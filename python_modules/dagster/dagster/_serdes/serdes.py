"""
Serialization & deserialization for Dagster objects.

Why have custom serialization?

* Default json serialization doesn't work well on namedtuples, which we use extensively to create
  immutable value types. Namedtuples serialize like tuples as flat lists.
* Explicit whitelisting should help ensure we are only persisting or communicating across a
  serialization boundary the types we expect to.

Why not pickle?

* This isn't meant to replace pickle in the conditions that pickle is reasonable to use
  (in memory, not human readable, etc) just handle the json case effectively.
"""

import collections.abc
from abc import ABC
from enum import Enum
from functools import lru_cache
from inspect import Parameter, signature
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
    overload,
)

from typing_extensions import Final, Literal, TypeAlias, TypeGuard, TypeVar

import dagster._check as check
import dagster._seven as seven
from dagster._utils import is_named_tuple_instance, is_named_tuple_subclass
from dagster._utils.cached_method import cached_method

from .errors import DeserializationError, SerdesUsageError, SerializationError

###################################################################################################
# Types
###################################################################################################

JsonSerializableValue: TypeAlias = Union[
    Sequence["JsonSerializableValue"],
    Mapping[str, "JsonSerializableValue"],
    str,
    int,
    float,
    bool,
    None,
]

PackableValue: TypeAlias = Union[
    Sequence["PackableValue"],
    Mapping[str, "PackableValue"],
    str,
    int,
    float,
    bool,
    None,
    NamedTuple,
    Set["PackableValue"],
    FrozenSet["PackableValue"],
    Enum,
]


###################################################################################################
# Whitelisting
###################################################################################################


class WhitelistMap(NamedTuple):
    tuples: Dict[str, "NamedTupleSerializer"]
    enums: Dict[str, "EnumSerializer"]
    nulls: Set[str]

    def register_tuple(
        self,
        name: str,
        named_tuple_class: Type[NamedTuple],
        serializer_class: Optional[Type["NamedTupleSerializer"]] = None,
        storage_name: Optional[str] = None,
        old_storage_names: Optional[AbstractSet[str]] = None,
        storage_field_names: Optional[Mapping[str, str]] = None,
        skip_when_empty_fields: Optional[AbstractSet[str]] = None,
    ):
        """Register a tuple in the whitelist map.

        Args:
            name: The class name of the namedtuple to register
            nt: The namedtuple class to register.
                Can be None to gracefull load previously serialized objects as None.
            serializer: The class to use when serializing and deserializing
        """
        serializer_class = serializer_class or NamedTupleSerializer
        serializer = serializer_class(
            klass=named_tuple_class,
            storage_name=storage_name,
            storage_field_names=storage_field_names,
            skip_when_empty_fields=skip_when_empty_fields,
        )
        self.tuples[name] = serializer
        if storage_name:
            self.tuples[storage_name] = serializer
        if old_storage_names:
            for old_storage_name in old_storage_names:
                self.tuples[old_storage_name] = serializer

    def has_tuple_entry(self, name: str) -> bool:
        return name in self.tuples

    def get_tuple_entry(self, name: str) -> "NamedTupleSerializer":
        return self.tuples[name]

    def register_enum(
        self,
        name: str,
        enum_class: Type[Enum],
        serializer_class: Optional[Type["EnumSerializer"]] = None,
        storage_name: Optional[str] = None,
        old_storage_names: Optional[AbstractSet[str]] = None,
    ) -> None:
        serializer_class = serializer_class or EnumSerializer
        serializer = serializer_class(
            klass=enum_class,
            storage_name=storage_name,
        )
        self.enums[name] = serializer
        if storage_name:
            self.enums[storage_name] = serializer
        if old_storage_names:
            for old_storage_name in old_storage_names:
                self.enums[old_storage_name] = serializer

    def has_enum_entry(self, name: str) -> bool:
        return name in self.enums

    def get_enum_entry(self, name: str) -> "EnumSerializer":
        return self.enums[name]

    def register_null(self, name: str) -> None:
        self.nulls.add(name)

    def has_null_entry(self, name: str) -> bool:
        return name in self.nulls

    @staticmethod
    def create() -> "WhitelistMap":
        return WhitelistMap(tuples={}, enums={}, nulls=set())


_WHITELIST_MAP: Final[WhitelistMap] = WhitelistMap.create()

T = TypeVar("T")
U = TypeVar("U")
T_Type = TypeVar("T_Type", bound=Type[object])
T_Scalar = TypeVar("T_Scalar", bound=Union[str, int, float, bool, None])


@overload
def whitelist_for_serdes(__cls: T_Type) -> T_Type:
    ...


@overload
def whitelist_for_serdes(
    __cls: None = None,
    *,
    serializer: Optional[Type["Serializer"]] = ...,
    storage_name: Optional[str] = ...,
    old_storage_names: Optional[AbstractSet[str]] = None,
    storage_field_names: Optional[Mapping[str, str]] = ...,
    skip_when_empty_fields: Optional[AbstractSet[str]] = ...,
) -> Callable[[T_Type], T_Type]:
    ...


def whitelist_for_serdes(
    __cls: Optional[T_Type] = None,
    *,
    serializer: Optional[Type["Serializer"]] = None,
    storage_name: Optional[str] = None,
    old_storage_names: Optional[AbstractSet[str]] = None,
    storage_field_names: Optional[Mapping[str, str]] = None,
    skip_when_empty_fields: Optional[AbstractSet[str]] = None,
) -> Union[T_Type, Callable[[T_Type], T_Type]]:
    """
    Decorator to whitelist a NamedTuple or enum to be serializable. If a `storage_name` is provided
    for a NamedTuple, then serialized instances of the NamedTuple will be stored under the
    `storage_name` instead of the class name. This is primarily useful for maintaining backwards
    compatibility. If a serialized object undergoes a name change, then setting `storage_name` to
    the old name will (a) allow the object to be deserialized by versions of Dagster prior to the
    name change; (b) allow Dagster to load objects stored using the old name.

    @whitelist_for_serdes
    class

    """
    if storage_field_names or skip_when_empty_fields:
        check.invariant(
            serializer is None or issubclass(serializer, NamedTupleSerializer),
            (
                "storage_field_names, skip_when_empty_fields can only be used with a"
                " NamedTupleSerializer"
            ),
        )
    if __cls is not None:  # decorator invoked directly on class
        check.class_param(__cls, "__cls")
        return _whitelist_for_serdes(whitelist_map=_WHITELIST_MAP)(__cls)
    else:  # decorator passed params
        check.opt_class_param(serializer, "serializer", superclass=Serializer)
        return _whitelist_for_serdes(
            whitelist_map=_WHITELIST_MAP,
            serializer=serializer,
            storage_name=storage_name,
            storage_field_names=storage_field_names,
            skip_when_empty_fields=skip_when_empty_fields,
        )


def _whitelist_for_serdes(
    whitelist_map: WhitelistMap,
    serializer: Optional[Type["Serializer"]] = None,
    storage_name: Optional[str] = None,
    old_storage_names: Optional[AbstractSet[str]] = None,
    storage_field_names: Optional[Mapping[str, str]] = None,
    skip_when_empty_fields: Optional[AbstractSet[str]] = None,
) -> Callable[[T_Type], T_Type]:
    def __whitelist_for_serdes(klass: T_Type) -> T_Type:
        if issubclass(klass, Enum) and (
            serializer is None or issubclass(serializer, EnumSerializer)
        ):
            whitelist_map.register_enum(
                klass.__name__,
                klass,
                serializer,
                storage_name=storage_name,
                old_storage_names=old_storage_names,
            )
            return klass
        elif is_named_tuple_subclass(klass) and (
            serializer is None or issubclass(serializer, NamedTupleSerializer)
        ):
            _check_serdes_tuple_class_invariants(klass)
            whitelist_map.register_tuple(
                klass.__name__,
                klass,
                serializer,
                storage_name=storage_name,
                old_storage_names=old_storage_names,
                storage_field_names=storage_field_names,
                skip_when_empty_fields=skip_when_empty_fields,
            )
            return klass  # type: ignore  # (NamedTuple quirk)
        else:
            raise SerdesUsageError(f"Can not whitelist class {klass} for serializer {serializer}")

    return __whitelist_for_serdes


class Serializer(ABC):
    pass


T_Enum = TypeVar("T_Enum", bound=Enum, default=Enum)


class EnumSerializer(Serializer, Generic[T_Enum]):
    def __init__(self, *, klass: Type[T_Enum], storage_name: Optional[str] = None):
        self.klass = klass
        self.storage_name = storage_name

    def unpack(self, value: str) -> T_Enum:
        return self.klass[value]

    def pack(self, value: Enum, whitelist_map: WhitelistMap, descent_path: str) -> str:
        return f"{self.get_storage_name()}.{value.name}"

    @lru_cache(maxsize=1)
    def get_storage_name(self) -> str:
        return self.storage_name or self.klass.__name__


T_NamedTuple = TypeVar("T_NamedTuple", bound=NamedTuple, default=NamedTuple)

EMPTY_VALUES_TO_SKIP: Tuple[None, List[Any], Dict[Any, Any], Set[Any]] = (None, [], {}, set())


class NamedTupleSerializer(Serializer, Generic[T_NamedTuple]):
    def __init__(
        self,
        *,
        klass: Type[T_NamedTuple],
        storage_name: Optional[str] = None,
        storage_field_names: Optional[Mapping[str, str]] = None,
        skip_when_empty_fields: Optional[AbstractSet[str]] = None,
        drop_undefined_fields: bool = True,
    ):
        self.klass = klass
        self.storage_name = storage_name
        self.storage_field_names = storage_field_names or {}
        self.skip_when_empty_fields = skip_when_empty_fields or set()

    def value_from_storage_dict(
        self,
        storage_dict: Dict[str, Any],
        whitelist_map: WhitelistMap,
        descent_path: str,
    ) -> T_NamedTuple:
        storage_dict = self.before_unpack(**storage_dict)
        unpacked: Dict[str, PackableValue] = {}
        for key, value in storage_dict.items():
            loaded_name = self.get_loaded_field_name(field=key)
            # Naively implements backwards compatibility by filtering arguments that aren't present in
            # the constructor. If a property is present in the serialized object, but doesn't exist in
            # the version of the class loaded into memory, that property will be completely ignored.
            if loaded_name in self.constructor_params:
                unpacked[loaded_name] = unpack_value(
                    value, whitelist_map=whitelist_map, descent_path=f"{descent_path}.{key}"
                )
        unpacked = self.after_unpack(**unpacked)

        # False positive type error here due to an eccentricity of `NamedTuple`-- calling `NamedTuple`
        # directly acts as a class factory, which is not true for `NamedTuple` subclasses (which act
        # like normal constructors). Because we have no subclass info here, the type checker thinks
        # we are invoking the class factory and complains about arguments.
        return self.klass(**unpacked)  # type: ignore

    # Hook: Modify the contents of the raw loaded dict before it is unpacked
    # into domain objects during deserialization.
    def before_unpack(self, **raw_dict: JsonSerializableValue) -> Dict[str, JsonSerializableValue]:
        return raw_dict

    def value_to_storage_dict(
        self,
        value: T_NamedTuple,
        whitelist_map: WhitelistMap,
        descent_path: str,
    ) -> Dict[str, JsonSerializableValue]:
        packed: Dict[str, JsonSerializableValue] = {}
        packed["__class__"] = self.get_storage_name()
        for key, inner_value in value._asdict().items():
            if key in self.skip_when_empty_fields and inner_value in EMPTY_VALUES_TO_SKIP:
                continue
            storage_key = self.get_storage_field_name(field=key)
            packed[storage_key] = pack_value(inner_value, whitelist_map, f"{descent_path}.{key}")
        packed = self.after_pack(**packed)
        return packed

    # Hook: Modify the contents of the dict after it is packed into a json
    # serializable form but before it is converted to a string.
    def after_pack(self, **packed_dict: JsonSerializableValue) -> Dict[str, JsonSerializableValue]:
        return packed_dict

    @property
    @lru_cache(maxsize=1)
    def constructor_params(self):
        return signature(self.klass.__new__).parameters

    @lru_cache(maxsize=1)
    def get_storage_name(self) -> str:
        return self.storage_name or self.klass.__name__

    @cached_method
    def get_storage_field_name(self, field: str) -> str:
        return self.storage_field_names.get(field, field)

    @cached_method
    def get_loaded_field_name(self, field: str) -> str:
        for k, v in self.storage_field_names.items():
            if v == field:
                return k
        return field


###################################################################################################
# Serialize
###################################################################################################


def serialize_value(
    val: PackableValue, whitelist_map: WhitelistMap = _WHITELIST_MAP, **json_kwargs: object
) -> str:
    """Serialize a whitelisted named tuple to a json encoded string."""
    packed_value = pack_value(val, whitelist_map=whitelist_map)
    return seven.json.dumps(packed_value, **json_kwargs)


@overload
def pack_value(
    val: T_Scalar, whitelist_map: WhitelistMap = ..., descent_path: Optional[str] = ...
) -> T_Scalar:
    ...


@overload
def pack_value(
    val: Union[
        Mapping[str, PackableValue], Set[PackableValue], FrozenSet[PackableValue], NamedTuple, Enum
    ],
    whitelist_map: WhitelistMap = ...,
    descent_path: Optional[str] = ...,
) -> Mapping[str, JsonSerializableValue]:
    ...


@overload
def pack_value(
    val: Sequence[PackableValue],
    whitelist_map: WhitelistMap = ...,
    descent_path: Optional[str] = ...,
) -> Sequence[JsonSerializableValue]:
    ...


def pack_value(
    val: PackableValue,
    whitelist_map: WhitelistMap = _WHITELIST_MAP,
    descent_path: Optional[str] = None,
) -> JsonSerializableValue:
    """
    Transform a value in to a json serializable form.

    The following types are transformed in to dicts:
        * whitelisted named tuples
        * whitelisted enums
        * set
        * frozenset
    """
    descent_path = _root(val) if descent_path is None else descent_path
    return _pack(val, whitelist_map=whitelist_map, descent_path=_root(val))


def _pack(
    val: PackableValue, whitelist_map: WhitelistMap, descent_path: str
) -> JsonSerializableValue:
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    if is_named_tuple_instance(val):
        klass_name = val.__class__.__name__
        if not whitelist_map.has_tuple_entry(klass_name):
            raise SerializationError(
                (
                    "Can only serialize whitelisted namedtuples, received"
                    f" {val}.{_path_msg(descent_path)}"
                ),
            )
        serializer = whitelist_map.get_tuple_entry(klass_name)
        return serializer.value_to_storage_dict(val, whitelist_map, descent_path)
    if isinstance(val, Enum):
        klass_name = val.__class__.__name__
        if not whitelist_map.has_enum_entry(klass_name):
            raise SerializationError(
                (
                    "Can only serialize whitelisted Enums, received"
                    f" {klass_name}.{_path_msg(descent_path)}"
                ),
            )
        enum_serializer = whitelist_map.get_enum_entry(klass_name)
        return {"__enum__": enum_serializer.pack(val, whitelist_map, descent_path)}
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val
    if isinstance(val, collections.abc.Sequence):
        return [
            _pack(item, whitelist_map, f"{descent_path}[{idx}]") for idx, item in enumerate(val)
        ]
    if isinstance(val, set):
        set_path = descent_path + "{}"
        return {
            "__set__": [_pack(item, whitelist_map, set_path) for item in sorted(list(val), key=str)]
        }
    if isinstance(val, frozenset):
        frz_set_path = descent_path + "{}"
        return {
            "__frozenset__": [
                _pack(item, whitelist_map, frz_set_path) for item in sorted(list(val), key=str)
            ]
        }
    if isinstance(val, collections.abc.Mapping):
        return {
            key: _pack(value, whitelist_map, f"{descent_path}.{key}") for key, value in val.items()
        }

    # list/dict checks above don't fully cover Sequence/Mapping
    return val


###################################################################################################
# Deserialize
###################################################################################################

T_PackableValue = TypeVar("T_PackableValue", bound=PackableValue, default=PackableValue)
U_PackableValue = TypeVar("U_PackableValue", bound=PackableValue, default=PackableValue)


@overload
def deserialize_value(
    val: str,
    as_type: Tuple[Type[T_PackableValue], Type[U_PackableValue]],
    whitelist_map: WhitelistMap = ...,
) -> Union[T_PackableValue, U_PackableValue]:
    ...


@overload
def deserialize_value(
    val: str,
    as_type: Type[T_PackableValue],
    whitelist_map: WhitelistMap = ...,
) -> T_PackableValue:
    ...


@overload
def deserialize_value(
    val: str,
    as_type: None = ...,
    whitelist_map: WhitelistMap = ...,
) -> PackableValue:
    ...


def deserialize_value(
    val: str,
    as_type: Optional[
        Union[Type[T_PackableValue], Tuple[Type[T_PackableValue], Type[U_PackableValue]]]
    ] = None,
    whitelist_map: WhitelistMap = _WHITELIST_MAP,
) -> Union[PackableValue, T_PackableValue, Union[T_PackableValue, U_PackableValue]]:
    """Deserialize a json encoded string to a Python object.

    Three steps:

    - Parse the input string as JSON.
    - Unpack the complex of lists, dicts, and scalars resulting from JSON parsing into a complex of richer
      Python objects (e.g. dagster-specific `NamedTuple` objects).
    - Optionally, check that the resulting object is of the expected type.
    """
    check.str_param(val, "val")
    packed_value = seven.json.loads(val)
    unpacked_value = unpack_value(packed_value, whitelist_map=whitelist_map)
    if as_type and not (
        is_named_tuple_instance(unpacked_value)
        if as_type is NamedTuple
        else isinstance(unpacked_value, as_type)
    ):
        raise DeserializationError(
            f"Deserialized object was not expected type {as_type}, got {type(unpacked_value)}"
        )
    return unpacked_value


@overload
def unpack_value(
    val: JsonSerializableValue,
    as_type: Tuple[Type[T_PackableValue], Type[U_PackableValue]],
    whitelist_map: WhitelistMap = ...,
    descent_path: str = ...,
) -> Union[T_PackableValue, U_PackableValue]:
    ...


@overload
def unpack_value(
    val: JsonSerializableValue,
    as_type: Type[T_PackableValue],
    whitelist_map: WhitelistMap = ...,
    descent_path: str = ...,
) -> T_PackableValue:
    ...


@overload
def unpack_value(
    val: JsonSerializableValue,
    as_type: None = ...,
    whitelist_map: WhitelistMap = ...,
    descent_path: str = ...,
) -> PackableValue:
    ...


def unpack_value(
    val: JsonSerializableValue,
    as_type: Optional[
        Union[Type[T_PackableValue], Tuple[Type[T_PackableValue], Type[U_PackableValue]]]
    ] = None,
    whitelist_map: WhitelistMap = _WHITELIST_MAP,
    descent_path: Optional[str] = None,
) -> Union[PackableValue, T_PackableValue, Union[T_PackableValue, U_PackableValue]]:
    """Convert a packed value in to its original form."""
    descent_path = _root(val) if descent_path is None else descent_path
    unpacked_value = _unpack(
        val,
        whitelist_map,
        descent_path,
    )
    if as_type and not (
        is_named_tuple_instance(unpacked_value)
        if as_type is NamedTuple
        else isinstance(unpacked_value, as_type)
    ):
        raise DeserializationError(
            f"Unpacked object was not expected type {as_type}, got {type(val)}"
        )
    return unpacked_value


def _unpack(
    val: JsonSerializableValue, whitelist_map: WhitelistMap, descent_path: str
) -> PackableValue:
    if isinstance(val, list):
        return [
            _unpack(item, whitelist_map, f"{descent_path}[{idx}]") for idx, item in enumerate(val)
        ]
    if isinstance(val, dict) and val.get("__class__"):
        klass_name = cast(str, val.pop("__class__"))
        if whitelist_map.has_null_entry(klass_name):
            return None
        elif not whitelist_map.has_tuple_entry(klass_name):
            raise DeserializationError(
                f"Attempted to deserialize class {klass_name} which is not in the whitelist. "
                "This error can occur due to version skew, verify processes are running "
                f"expected versions.{_path_msg(descent_path)}"
            )

        serializer = whitelist_map.get_tuple_entry(klass_name)
        return serializer.value_from_storage_dict(val, whitelist_map, descent_path)
    if isinstance(val, dict) and val.get("__enum__"):
        enum = cast(str, val["__enum__"])
        name, member = enum.split(".")
        if not whitelist_map.has_enum_entry(name):
            raise DeserializationError(
                f"Attempted to deserialize enum {name} which was not in the whitelist.\n"
                "This error can occur due to version skew, verify processes are running "
                f"expected versions.{_path_msg(descent_path)}"
            )
        enum_serializer = whitelist_map.get_enum_entry(name)
        return enum_serializer.unpack(member)
    if isinstance(val, dict) and "__set__" in val:
        set_path = descent_path + "{}"
        items = cast(List[JsonSerializableValue], val["__set__"])
        return set([_unpack(item, whitelist_map, set_path) for item in items])
    if isinstance(val, dict) and "__frozenset__" in val:
        frz_set_path = descent_path + "{}"
        items = cast(List[JsonSerializableValue], val["__frozenset__"])
        return frozenset([_unpack(item, whitelist_map, frz_set_path) for item in items])
    if isinstance(val, dict):
        return {
            key: _unpack(value, whitelist_map, f"{descent_path}.{key}")
            for key, value in val.items()
        }

    return val


###################################################################################################
# Back compat
###################################################################################################


def register_serdes_null_deserialization(
    *names: str, whitelist_map: WhitelistMap = _WHITELIST_MAP
) -> None:
    """
    Manually provide remappings for serialized records.
    Used to load types that no longer exist as None.
    """
    for name in names:
        whitelist_map.register_null(name)


###################################################################################################
# Validation
###################################################################################################


def _check_serdes_tuple_class_invariants(klass: Type[NamedTuple]) -> None:
    sig_params = signature(klass.__new__).parameters
    dunder_new_params = list(sig_params.values())

    cls_param = dunder_new_params[0]

    def _with_header(msg: str) -> str:
        return f"For namedtuple {klass.__name__}: {msg}"

    if cls_param.name not in {"cls", "_cls"}:
        raise SerdesUsageError(
            _with_header(f'First parameter must be _cls or cls. Got "{cls_param.name}".')
        )

    value_params = dunder_new_params[1:]

    for index, field in enumerate(klass._fields):
        if index >= len(value_params):
            error_msg = (
                "Missing parameters to __new__. You have declared fields "
                "in the named tuple that are not present as parameters to the "
                "to the __new__ method. In order for "
                "both serdes serialization and pickling to work, "
                "these must match. Missing: {missing_fields}"
            ).format(missing_fields=repr(list(klass._fields[index:])))

            raise SerdesUsageError(_with_header(error_msg))

        value_param = value_params[index]
        if value_param.name != field:
            error_msg = (
                "Params to __new__ must match the order of field declaration in the namedtuple. "
                'Declared field number {one_based_index} in the namedtuple is "{field_name}". '
                'Parameter {one_based_index} in __new__ method is "{param_name}".'
            ).format(one_based_index=index + 1, field_name=field, param_name=value_param.name)
            raise SerdesUsageError(_with_header(error_msg))

    if len(value_params) > len(klass._fields):
        # Ensure that remaining parameters have default values
        for extra_param_index in range(len(klass._fields), len(value_params) - 1):
            if value_params[extra_param_index].default == Parameter.empty:
                error_msg = (
                    'Parameter "{param_name}" is a parameter to the __new__ '
                    "method but is not a field in this namedtuple. The only "
                    "reason why this should exist is that "
                    "it is a field that used to exist (we refer to this as the graveyard) "
                    "but no longer does. However it might exist in historical storage. This "
                    "parameter existing ensures that serdes continues to work. However these "
                    "must come at the end and have a default value for pickling to work."
                ).format(param_name=value_params[extra_param_index].name)
                raise SerdesUsageError(_with_header(error_msg))


def _path_msg(descent_path: str) -> str:
    if not descent_path:
        return ""
    else:
        return f"\nDescent path: {descent_path}"


def _root(val: Any) -> str:
    return f"<root:{val.__class__.__name__}>"


def is_packed_enum(val: object) -> TypeGuard[Mapping[str, str]]:
    return isinstance(val, dict) and "__enum__" in val


def copy_packed_set(
    packed_set: Mapping[str, Sequence[JsonSerializableValue]],
    as_type: Literal["__frozenset__", "__set__"],
) -> Mapping[str, Sequence[JsonSerializableValue]]:
    """
    Returns a copy of the packed collection
    """
    if "__set__" in packed_set:
        return {as_type: packed_set["__set__"]}
    elif "__frozenset__" in packed_set:
        return {as_type: packed_set["__frozenset__"]}
    else:
        check.failed(f"Invalid packed set: {packed_set}")
