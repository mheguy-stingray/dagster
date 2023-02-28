Databricks (dagster-databricks)
-------------------------------

The ``dagster_databricks`` package provides the main pieces of functionality:

- A resource, ``databricks_pyspark_step_launcher``, which will execute a op within a Databricks
  context on a cluster, such that the ``pyspark`` resource uses the cluster's Spark instance.

Note that, for the ``databricks_pyspark_step_launcher``, either S3 or Azure Data Lake Storage config
**must** be specified for ops to succeed, and the credentials for this storage must also be
stored as a Databricks Secret and stored in the resource config so that the Databricks cluster can
access storage.

APIs
----
.. currentmodule:: dagster_databricks

.. autofunction:: dagster_databricks.create_databricks_run_now_op

.. autofunction:: dagster_databricks.create_databricks_submit_run_op

.. autoconfigurable:: dagster_databricks.databricks_pyspark_step_launcher
  :annotation: ResourceDefinition

.. autoclass:: dagster_databricks.DatabricksError

Resources
=========

.. autoclass:: DatabricksClient
  :members:

.. autoconfigurable:: databricks_client
  :annotation: ResourceDefinition
