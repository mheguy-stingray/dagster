from typing import Optional, Sequence

import pytest
from dagster import In, Nothing, Out, job
from dagster._check import CheckError
from dagster_databricks import databricks_client
from dagster_databricks.ops import (
    create_databricks_run_now_op,
    create_databricks_submit_run_op,
)
from dagster_databricks.types import (
    DatabricksRunLifeCycleState,
    DatabricksRunResultState,
)
from pytest_mock import MockerFixture


def _mock_get_run_response() -> Sequence[dict]:
    return [
        {
            "run_name": "my_databricks_run",
            "run_page_url": "https://abc123.cloud.databricks.com/?o=12345#job/1/run/1",
        },
        {
            "state": {
                "life_cycle_state": DatabricksRunLifeCycleState.PENDING,
                "state_message": "",
            }
        },
        {
            "state": {
                "life_cycle_state": DatabricksRunLifeCycleState.RUNNING,
                "state_message": "",
            }
        },
        {
            "state": {
                "result_state": DatabricksRunResultState.SUCCESS,
                "life_cycle_state": DatabricksRunLifeCycleState.TERMINATED,
                "state_message": "Finished",
            }
        },
    ]


@pytest.mark.parametrize(
    "op_kwargs",
    [
        {},
        {
            "ins": {"input1": In(Nothing), "input2": In(Nothing)},
            "out": {
                "output1": Out(Nothing, is_required=False),
                "output2": Out(Nothing, is_required=False),
            },
        },
    ],
    ids=[
        "no overrides",
        "with ins and outs",
    ],
)
@pytest.mark.parametrize(
    "databricks_job_configuration",
    [
        None,
        {},
        {
            "python_params": [
                "--input",
                "schema.db.input_table",
                "--output",
                "schema.db.output_table",
            ]
        },
    ],
    ids=[
        "no Databricks job configuration",
        "empty Databricks job configuration",
        "Databricks job configuration with python params",
    ],
)
def test_databricks_run_now_op(
    mocker: MockerFixture, op_kwargs: dict, databricks_job_configuration: Optional[dict]
) -> None:
    mock_run_now = mocker.patch("databricks_cli.sdk.JobsService.run_now")
    mock_get_run = mocker.patch("databricks_cli.sdk.JobsService.get_run")
    databricks_job_id = 10

    mock_run_now.return_value = {"run_id": 1}
    mock_get_run.side_effect = _mock_get_run_response()

    test_databricks_run_now_op = create_databricks_run_now_op(
        databricks_job_id=databricks_job_id,
        databricks_job_configuration=databricks_job_configuration,
        **op_kwargs,
        poll_interval_seconds=0.01,
    )

    @job(
        resource_defs={
            "databricks": databricks_client.configured(
                {"host": "https://abc123.cloud.databricks.com/", "token": "token"}
            )
        }
    )
    def test_databricks_job() -> None:
        test_databricks_run_now_op()

    result = test_databricks_job.execute_in_process()

    assert result.success
    mock_run_now.assert_called_once_with(
        job_id=databricks_job_id,
        **(databricks_job_configuration or {}),
    )
    assert mock_get_run.call_count == 4


@pytest.mark.parametrize(
    "op_kwargs",
    [
        {},
        {
            "ins": {"input1": In(Nothing), "input2": In(Nothing)},
            "out": {
                "output1": Out(Nothing, is_required=False),
                "output2": Out(Nothing, is_required=False),
            },
        },
    ],
    ids=[
        "no overrides",
        "with ins and outs",
    ],
)
def test_databricks_submit_run_op(mocker: MockerFixture, op_kwargs: dict) -> None:
    mock_submit_run = mocker.patch("databricks_cli.sdk.JobsService.submit_run")
    mock_get_run = mocker.patch("databricks_cli.sdk.JobsService.get_run")

    mock_submit_run.return_value = {"run_id": 1}
    mock_get_run.side_effect = _mock_get_run_response()

    test_databricks_submit_run_op = create_databricks_submit_run_op(
        databricks_job_configuration={
            "new_cluster": {
                "spark_version": "2.1.0-db3-scala2.11",
                "num_workers": 2,
            },
            "notebook_task": {
                "notebook_path": "/Users/dagster@example.com/PrepareData",
            },
        },
        **op_kwargs,
        poll_interval_seconds=0.01,
    )

    @job(
        resource_defs={
            "databricks": databricks_client.configured(
                {"host": "https://abc123.cloud.databricks.com/", "token": "token"}
            )
        }
    )
    def test_databricks_job() -> None:
        test_databricks_submit_run_op()

    result = test_databricks_job.execute_in_process()

    assert result.success
    assert mock_submit_run.call_count == 1
    assert mock_get_run.call_count == 4


def test_databricks_submit_run_op_no_job() -> None:
    with pytest.raises(CheckError):
        create_databricks_submit_run_op(databricks_job_configuration={})
