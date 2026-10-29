# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Ingest regulatory and academic PDFs into UC volumes and mra_source_documents."""
# COMMAND ----------
# MAGIC %md
# MAGIC # Ingest data
# MAGIC
# MAGIC Job notebook: loads `project_config.yml` and `regulatory_sources.yml`, then downloads
# MAGIC regulatory and academic PDFs via `DataProcessor` into your schema volumes / tables.
# MAGIC The `model_risk_agent` wheel is installed by the job environment.

# COMMAND ----------

from databricks.sdk.runtime import dbutils
from loguru import logger
from pyspark.sql import SparkSession

from model_risk_agent.config import load_config, load_regulatory_sources
from model_risk_agent.data_processor import DataProcessor

# The environment comes from the job's base parameters; "dev" when run by hand.
dbutils.widgets.text("env", "dev")
env = dbutils.widgets.get("env")

cfg = load_config("../../project_config.yml", env=env)
logger.info(f"Configuration loaded for {cfg.full_schema_name}")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

regulatory_sources = load_regulatory_sources("../../regulatory_sources.yml")
logger.info(f"Loaded {len(regulatory_sources)} regulatory sources")

# COMMAND ----------

data_processor = DataProcessor(spark, cfg)
data_processor.download_and_store_public_pdfs(regulatory_sources)

# COMMAND ----------

data_processor.download_and_store_papers()
