# Databricks notebook source

from datetime import datetime

import arxiv
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, LongType, StringType, StructField, StructType

from model_risk_agent.config import get_env, load_config, load_regulatory_sources
from model_risk_agent.data_processor import DataProcessor

# COMMAND ----------
# Create Spark session
spark = SparkSession.builder.getOrCreate()

# Load config
env = get_env()
cfg = load_config("../project_config.yml", env)


# COMMAND ----------
# Fetch arXiv Paper Metadata from arXiv API: https://arxiv.org/help/api/index

regulatory_sources = load_regulatory_sources("../regulatory_sources.yml")
print(regulatory_sources)


# COMMAND ----------
# Create Delta Table in Unity Catalog

data_processor = DataProcessor(spark, cfg)
data_processor.download_and_store_public_pdfs(regulatory_sources)

# COMMAND ----------
# Verify the Data

# Read back the table
papers_df = spark.table(f"{CATALOG}.{SCHEMA}.{TABLE_NAME}")

logger.info(f"Table: {CATALOG}.{SCHEMA}.{TABLE_NAME}")
logger.info(f"Total papers: {papers_df.count()}")
logger.info("Schema:")
papers_df.printSchema()

logger.info("Sample records:")
papers_df.select("arxiv_id", "title", "primary_category", "published").show(5, truncate=50)

# COMMAND ----------
# Data Statistics

logger.info("Papers by primary category:")
papers_df.groupBy("primary_category").count().orderBy("count", ascending=False).show()

logger.info("Most recent papers:")
papers_df.select("title", "published", "arxiv_id").orderBy("published", ascending=False).show(
    5, truncate=60
)
