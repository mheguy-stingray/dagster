import tempfile

import pytest
from click.testing import CliRunner
from dagster import AssetKey, AssetMaterialization, Output
from dagster._cli.asset import asset_wipe_command
from dagster._core.definitions import op
from dagster._core.test_utils import instance_for_test
from dagster._legacy import execute_pipeline, pipeline
from dagster._seven import json


@pytest.fixture(name="instance_runner")
def mock_instance_runner():
    with tempfile.TemporaryDirectory() as dagster_home_temp:
        with instance_for_test(
            temp_dir=dagster_home_temp,
        ) as instance:
            runner = CliRunner(env={"DAGSTER_HOME": dagster_home_temp})
            yield instance, runner


@op
def solid_one(_):
    yield AssetMaterialization(asset_key=AssetKey("asset_1"))
    yield Output(1)


@op
def solid_two(_):
    yield AssetMaterialization(asset_key=AssetKey("asset_2"))
    yield AssetMaterialization(asset_key=AssetKey(["path", "to", "asset_3"]))
    yield AssetMaterialization(asset_key=AssetKey(("path", "to", "asset_4")))
    yield Output(1)


@op
def solid_normalization(_):
    yield AssetMaterialization(asset_key="path/to-asset_5")
    yield Output(1)


@pipeline
def pipeline_one():
    solid_one()


@pipeline
def pipeline_two():
    solid_one()
    solid_two()


def test_asset_wipe_errors(instance_runner):  # pylint: disable=unused-argument
    _, runner = instance_runner
    result = runner.invoke(asset_wipe_command)
    assert result.exit_code == 2
    assert (
        "Error, you must specify an asset key or use `--all` to wipe all asset keys."
        in result.output
    )

    result = runner.invoke(asset_wipe_command, ["--all", json.dumps(["path", "to", "asset_key"])])
    assert result.exit_code == 2
    assert "Error, cannot use more than one of: asset key, `--all`." in result.output


def test_asset_exit(instance_runner):  # pylint: disable=unused-argument
    _, runner = instance_runner
    result = runner.invoke(asset_wipe_command, ["--all"], input="NOT_DELETE\n")
    assert result.exit_code == 0
    assert "Exiting without removing asset indexes" in result.output


def test_asset_single_wipe(instance_runner):
    instance, runner = instance_runner
    execute_pipeline(pipeline_one, instance=instance)
    execute_pipeline(pipeline_two, instance=instance)
    asset_keys = instance.all_asset_keys()
    assert len(asset_keys) == 4

    result = runner.invoke(
        asset_wipe_command, [json.dumps(["path", "to", "asset_3"])], input="DELETE\n"
    )
    assert result.exit_code == 0
    assert "Removed asset indexes from event logs" in result.output

    result = runner.invoke(
        asset_wipe_command, [json.dumps(["path", "to", "asset_4"])], input="DELETE\n"
    )
    assert result.exit_code == 0
    assert "Removed asset indexes from event logs" in result.output

    asset_keys = instance.all_asset_keys()
    assert len(asset_keys) == 2


def test_asset_multi_wipe(instance_runner):
    instance, runner = instance_runner
    execute_pipeline(pipeline_one, instance=instance)
    execute_pipeline(pipeline_two, instance=instance)
    asset_keys = instance.all_asset_keys()
    assert len(asset_keys) == 4

    result = runner.invoke(
        asset_wipe_command,
        [json.dumps(["path", "to", "asset_3"]), json.dumps(["asset_1"])],
        input="DELETE\n",
    )
    assert result.exit_code == 0
    assert "Removed asset indexes from event logs" in result.output
    asset_keys = instance.all_asset_keys()
    assert len(asset_keys) == 2


def test_asset_wipe_all(instance_runner):
    instance, runner = instance_runner
    execute_pipeline(pipeline_one, instance=instance)
    execute_pipeline(pipeline_two, instance=instance)
    asset_keys = instance.all_asset_keys()
    assert len(asset_keys) == 4

    result = runner.invoke(asset_wipe_command, ["--all"], input="DELETE\n")
    assert result.exit_code == 0
    assert "Removed asset indexes from event logs" in result.output
    asset_keys = instance.all_asset_keys()
    assert len(asset_keys) == 0


def test_asset_single_wipe_noprompt(instance_runner):
    instance, runner = instance_runner
    execute_pipeline(pipeline_one, instance=instance)
    execute_pipeline(pipeline_two, instance=instance)
    asset_keys = instance.all_asset_keys()
    assert len(asset_keys) == 4

    result = runner.invoke(
        asset_wipe_command, ["--noprompt", json.dumps(["path", "to", "asset_3"])]
    )
    assert result.exit_code == 0
    assert "Removed asset indexes from event logs" in result.output

    asset_keys = instance.all_asset_keys()
    assert len(asset_keys) == 3
