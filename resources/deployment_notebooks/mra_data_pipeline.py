# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Run the MRA data pipeline: download, parse, chunk, and embed into UC tables."""
# COMMAND ----------
# MAGIC %md
# MAGIC # MRA data pipeline
# MAGIC
# MAGIC Job notebook: loads `project_config.yml`, then runs `DataProcessor.process_and_save()`
# MAGIC (regulatory + academic download → AI parse → chunks + embeddings).
# MAGIC The `model_risk_agent` wheel is installed by the job environment.

# COMMAND ----------

from databricks.sdk.runtime import dbutils
from loguru import logger
from pyspark.sql import SparkSession

from model_risk_agent.config import load_config
from model_risk_agent.data_processor import DataProcessor

# The environment comes from the job's base parameters; "dev" when run by hand.
dbutils.widgets.text("env", "dev")
env = dbutils.widgets.get("env")

cfg = load_config("../../project_config.yml", env=env)
logger.info(f"Configuration loaded for {cfg.full_schema_name}")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

processor = DataProcessor(spark, cfg)
processor.process_and_save(regulatory_sources_path="../../regulatory_sources.yml")
