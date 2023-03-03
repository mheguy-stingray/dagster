import datetime
from typing import cast

import pytest
from dagster import (
    DagsterInstance,
    DagsterInvariantViolationError,
    RunRequest,
    build_schedule_context,
    schedule,
)
from dagster._config.structured_config import ConfigurableResource
from dagster._core.errors import DagsterInvalidInvocationError
from dagster._core.test_utils import instance_for_test
from dagster._legacy import daily_schedule


def cron_test_schedule_factory_context():
    @schedule(cron_schedule="* * * * *", job_name="no_pipeline")
    def basic_schedule(_):
        return {}

    return basic_schedule


def cron_test_schedule_factory_no_context():
    @schedule(cron_schedule="* * * * *", job_name="no_pipeline")
    def basic_schedule():
        return {}

    return basic_schedule


def test_cron_schedule_invocation_all_args():
    basic_schedule_context = cron_test_schedule_factory_context()

    assert basic_schedule_context(None) == {}
    assert basic_schedule_context(build_schedule_context()) == {}
    assert basic_schedule_context(_=None) == {}
    assert basic_schedule_context(_=build_schedule_context()) == {}

    basic_schedule_no_context = cron_test_schedule_factory_no_context()

    assert basic_schedule_no_context() == {}


def test_incorrect_cron_schedule_invocation():
    basic_schedule = cron_test_schedule_factory_context()

    with pytest.raises(
        DagsterInvalidInvocationError,
        match=(
            "Schedule decorated function has context argument, but no context argument was "
            "provided."
        ),
    ):
        basic_schedule()  # pylint: disable=no-value-for-parameter

    with pytest.raises(
        DagsterInvalidInvocationError,
        match="Schedule invocation expected argument '_'.",
    ):
        basic_schedule(foo=None)  # pylint: disable=no-value-for-parameter,unexpected-keyword-arg


def partition_schedule_factory():
    @daily_schedule(
        pipeline_name="test_pipeline",
        start_date=datetime.datetime(2020, 1, 1),
    )
    def my_partition_schedule(date):
        assert isinstance(date, datetime.datetime)
        return {}

    return my_partition_schedule


def test_partition_schedule_invocation_all_args():
    my_partition_schedule = partition_schedule_factory()
    test_date = datetime.datetime(2020, 1, 1)
    assert my_partition_schedule(test_date) == {}
    assert my_partition_schedule(date=test_date) == {}


def test_incorrect_partition_schedule_invocation():
    my_partition_schedule = partition_schedule_factory()
    with pytest.raises(
        DagsterInvalidInvocationError,
        match="Schedule decorated function has date argument, but no date argument was provided.",
    ):
        my_partition_schedule()  # pylint: disable=no-value-for-parameter

    with pytest.raises(
        DagsterInvalidInvocationError,
        match="Schedule invocation expected argument 'date'.",
    ):
        my_partition_schedule(  # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
            foo=None
        )


def test_instance_access():
    with pytest.raises(
        DagsterInvariantViolationError,
        match="Attempted to initialize dagster instance, but no instance reference was provided.",
    ):
        build_schedule_context().instance  # pylint: disable=expression-not-assigned

    with instance_for_test() as instance:
        assert isinstance(build_schedule_context(instance).instance, DagsterInstance)


def test_schedule_invocation_resources() -> None:
    class MyResource(ConfigurableResource):
        a_str: str

    # Test no arg invocation
    @schedule(job_name="foo_pipeline", cron_schedule="* * * * *")
    def basic_schedule_resource_req(my_resource: MyResource):
        return RunRequest(run_key=None, run_config={"foo": my_resource.a_str}, tags={})

    # with pytest.raises(CheckError, match="Schedule missing required resources: my_resource"):
    #     basic_schedule_resource_req()

    assert hasattr(
        build_schedule_context(resource_defs={"my_resource": MyResource(a_str="foo")}).resources,
        "my_resource",
    )

    # Just need to pass context, which splats out into resource parameters
    assert cast(
        RunRequest,
        basic_schedule_resource_req(
            build_schedule_context(resource_defs={"my_resource": MyResource(a_str="foo")})
        ),
    ).run_config == {"foo": "foo"}
