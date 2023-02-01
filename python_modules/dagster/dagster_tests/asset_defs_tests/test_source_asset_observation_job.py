from typing import Optional

import pytest
from dagster._core.definitions.decorators.source_asset_decorator import observable_source_asset
from dagster._core.definitions.definitions_class import Definitions
from dagster._core.definitions.events import AssetKey
from dagster._core.definitions.logical_version import (
    LogicalVersion,
    extract_logical_version_from_entry,
)
from dagster._core.definitions.resource_definition import ResourceDefinition
from dagster._core.definitions.unresolved_asset_job_definition import define_asset_job
from dagster._core.errors import DagsterInvalidDefinitionError
from dagster._core.instance import DagsterInstance


def _get_current_logical_version(
    key: AssetKey, instance: DagsterInstance
) -> Optional[LogicalVersion]:
    record = instance.get_latest_logical_version_record(key)
    assert record is not None
    return extract_logical_version_from_entry(record.event_log_entry)


def test_execute_source_asset_observation_job():
    executed = {}

    @observable_source_asset
    def foo(_context) -> LogicalVersion:
        executed["foo"] = True
        return LogicalVersion("alpha")

    @observable_source_asset
    def bar(context):
        executed["bar"] = True
        return LogicalVersion("beta")

    instance = DagsterInstance.ephemeral()

    result = (
        Definitions(
            assets=[foo, bar],
            jobs=[define_asset_job("source_asset_job", [foo, bar])],
        )
        .get_job_def("source_asset_job")
        .execute_in_process(instance=instance)
    )

    assert result.success
    assert executed["foo"]
    assert _get_current_logical_version(AssetKey("foo"), instance) == LogicalVersion("alpha")
    assert executed["bar"]
    assert _get_current_logical_version(AssetKey("bar"), instance) == LogicalVersion("beta")


@pytest.mark.parametrize(
    "is_valid,resource_defs",
    [(True, {"bar": ResourceDefinition.hardcoded_resource("bar")}), (False, {})],
)
def test_source_asset_observation_job_with_resource(is_valid, resource_defs):
    executed = {}

    @observable_source_asset(
        required_resource_keys={"bar"},
    )
    def foo(context) -> LogicalVersion:
        executed["foo"] = True
        return LogicalVersion(f"{context.resources.bar}")

    instance = DagsterInstance.ephemeral()

    if is_valid:
        print("RESOURCE DEFS", resource_defs)
        result = (
            Definitions(
                assets=[foo],
                jobs=[define_asset_job("source_asset_job", [foo])],
                resources=resource_defs,
            )
            .get_job_def("source_asset_job")
            .execute_in_process(instance=instance)
        )

        assert result.success
        assert executed["foo"]
        assert _get_current_logical_version(AssetKey("foo"), instance) == LogicalVersion("bar")
    else:
        with pytest.raises(
            DagsterInvalidDefinitionError,
            match="resource with key 'bar' required by op 'foo' was not provided",
        ):
            result = (
                Definitions(
                    assets=[foo],
                    jobs=[define_asset_job("source_asset_job", [foo])],
                    resources=resource_defs,
                )
                .get_job_def("source_asset_job")
                .execute_in_process(instance=instance)
            )
