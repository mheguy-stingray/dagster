# pyright: reportGeneralTypeIssues=none

# Disable reportGeneralTypeIssues here due to use of Dagster types in annotations

import typing

import pytest
import yaml

from dagster import (
    AssetMaterialization,
    DagsterType,
    DagsterTypeCheckDidNotPass,
    In,
    Nothing,
    Out,
    PythonObjectDagsterType,
    dagster_type_loader,
    make_python_type_usable_as_dagster_type,
    op,
    usable_as_dagster_type,
)
from dagster._legacy import execute_pipeline, execute_solid, pipeline
from dagster._utils import safe_tempfile_path


def test_basic_even_type():
    # start_test_basic_even_type
    EvenDagsterType = DagsterType(
        name="EvenDagsterType",
        type_check_fn=lambda _, value: isinstance(value, int) and value % 2 == 0,
    )
    # end_test_basic_even_type

    # start_test_basic_even_type_with_annotations
    @op
    def double_even(num: EvenDagsterType) -> EvenDagsterType:
        # These type annotations are a shorthand for constructing InputDefinitions
        # and OutputDefinitions, and are not mypy compliant
        return num  # at runtime this is a python int

    # end_test_basic_even_type_with_annotations

    assert execute_solid(double_even, input_values={"num": 2}).success

    with pytest.raises(DagsterTypeCheckDidNotPass):
        execute_solid(double_even, input_values={"num": 3})

    assert not execute_solid(double_even, input_values={"num": 3}, raise_on_error=False).success


def test_basic_even_type_no_annotations():
    EvenDagsterType = DagsterType(
        name="EvenDagsterType",
        type_check_fn=lambda _, value: isinstance(value, int) and value % 2 == 0,
    )

    # start_test_basic_even_type_no_annotations
    @op(
        ins={"num": In(EvenDagsterType)},
        out=Out(EvenDagsterType),
    )
    def double_even(num):
        return num

    # end_test_basic_even_type_no_annotations

    assert execute_solid(double_even, input_values={"num": 2}).success

    with pytest.raises(DagsterTypeCheckDidNotPass):
        execute_solid(double_even, input_values={"num": 3})

    assert not execute_solid(double_even, input_values={"num": 3}, raise_on_error=False).success


def test_python_object_dagster_type():
    # start_object_type
    class EvenType:
        def __init__(self, num):
            assert num % 2 == 0
            self.num = num

    EvenDagsterType = PythonObjectDagsterType(EvenType, name="EvenDagsterType")
    # end_object_type

    # start_use_object_type
    @op
    def double_even(even_num: EvenDagsterType) -> EvenDagsterType:
        # These type annotations are a shorthand for constructing InputDefinitions
        # and OutputDefinitions, and are not mypy compliant
        return EvenType(even_num.num * 2)

    # end_use_object_type

    assert execute_solid(double_even, input_values={"even_num": EvenType(2)}).success
    with pytest.raises(AssertionError):
        execute_solid(double_even, input_values={"even_num": EvenType(3)})


def test_even_type_loader():
    # start_type_loader
    class EvenType:
        def __init__(self, num):
            assert num % 2 == 0
            self.num = num

    @dagster_type_loader(int)
    def load_even_type(_, cfg):
        return EvenType(cfg)

    EvenDagsterType = PythonObjectDagsterType(EvenType, loader=load_even_type)
    # end_type_loader

    @op
    def double_even(even_num: EvenDagsterType) -> EvenDagsterType:
        return EvenType(even_num.num * 2)

    # start_via_config
    yaml_doc = """
    ops:
        double_even:
            inputs:
                even_num: 2
    """
    # end_via_config
    assert execute_solid(double_even, run_config=yaml.safe_load(yaml_doc)).success

    assert execute_solid(
        double_even, run_config={"ops": {"double_even": {"inputs": {"even_num": 2}}}}
    ).success

    # Same same as above w/r/t chatting to prha
    with pytest.raises(AssertionError):
        execute_solid(
            double_even,
            run_config={"solids": {"double_even": {"inputs": {"even_num": 3}}}},
        )


def test_mypy_compliance():
    # start_mypy
    class EvenType:
        def __init__(self, num):
            assert num % 2 == 0
            self.num = num

    if typing.TYPE_CHECKING:
        EvenDagsterType = EvenType
    else:
        EvenDagsterType = PythonObjectDagsterType(EvenType)

    @op
    def double_even(even_num: EvenDagsterType) -> EvenDagsterType:
        return EvenType(even_num.num * 2)

    # end_mypy
    assert execute_solid(double_even, input_values={"even_num": EvenType(2)}).success


def test_nothing_type():
    @op(out={"cleanup_done": Out(Nothing)})
    def do_cleanup():
        pass

    @op(ins={"on_cleanup_done": In(Nothing)})
    def after_cleanup():  # Argument not required for Nothing types
        return "worked"

    @pipeline
    def nothing_pipeline():
        after_cleanup(do_cleanup())

    result = execute_pipeline(nothing_pipeline)
    assert result.success
    assert result.output_for_node("after_cleanup") == "worked"


def test_nothing_fanin_actually_test():
    ordering = {"counter": 0}

    @op(out=Out(Nothing))
    def start_first_pipeline_section(context):
        ordering["counter"] += 1
        ordering[context.op.name] = ordering["counter"]

    @op(
        ins={"first_section_done": In(Nothing)},
        out=Out(Nothing),
    )
    def perform_clean_up(context):
        ordering["counter"] += 1
        ordering[context.op.name] = ordering["counter"]

    @op(ins={"on_cleanup_tasks_done": In(Nothing)})
    def start_next_pipeline_section(context):
        ordering["counter"] += 1
        ordering[context.op.name] = ordering["counter"]
        return "worked"

    @pipeline
    def fanin_pipeline():
        first_section_done = start_first_pipeline_section()
        start_next_pipeline_section(
            on_cleanup_tasks_done=[
                perform_clean_up.alias("cleanup_task_one")(first_section_done),
                perform_clean_up.alias("cleanup_task_two")(first_section_done),
            ]
        )

    result = execute_pipeline(fanin_pipeline)
    assert result.success

    assert ordering["start_first_pipeline_section"] == 1
    assert ordering["start_next_pipeline_section"] == 4


def test_nothing_fanin_empty_body_for_guide():
    @op(out=Out(Nothing))
    def start_first_pipeline_section():
        pass

    @op(
        ins={"first_section_done": In(Nothing)},
        out=Out(Nothing),
    )
    def perform_clean_up():
        pass

    @op(
        ins={"on_cleanup_tasks_done": In(Nothing)},
    )
    def start_next_pipeline_section():
        pass

    @pipeline
    def fanin_pipeline():
        first_section_done = start_first_pipeline_section()
        start_next_pipeline_section(
            on_cleanup_tasks_done=[
                perform_clean_up.alias("cleanup_task_one")(first_section_done),
                perform_clean_up.alias("cleanup_task_two")(first_section_done),
            ]
        )

    result = execute_pipeline(fanin_pipeline)
    assert result.success


def test_usable_as_dagster_type():
    # start_usable_as
    @usable_as_dagster_type
    class EvenType:
        def __init__(self, num):
            assert num % 2 == 0
            self.num = num

    # end_usable_as
    @op
    def double_even(even_num: EvenType) -> EvenType:
        return EvenType(even_num.num * 2)

    assert execute_solid(double_even, input_values={"even_num": EvenType(2)}).success


def test_make_usable_as_dagster_type():
    # start_make_usable
    class EvenType:
        def __init__(self, num):
            assert num % 2 == 0
            self.num = num

    EvenDagsterType = PythonObjectDagsterType(
        EvenType,
        name="EvenDagsterType",
    )

    make_python_type_usable_as_dagster_type(EvenType, EvenDagsterType)

    @op
    def double_even(even_num: EvenType) -> EvenType:
        return EvenType(even_num.num * 2)

    # end_make_usable
    assert execute_solid(double_even, input_values={"even_num": EvenType(2)}).success
