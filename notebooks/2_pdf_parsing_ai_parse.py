# Databricks notebook source
# COMMAND ----------

# MAGIC %pip install --force-reinstall /Workspace/Users/lennon.jcruz@gmail.com/.bundle/dev/llmops-lennon-jcruz/files/notebooks/model_risk_agent-0.0.1-py3-none-any.whl
# COMMAND ----------
# MAGIC %md
# MAGIC # Lecture 2.2: PDF Parsing with AI Parse Documents
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Downloading and storing PDFs
# MAGIC - AI Parse Documents for intelligent parsing
# MAGIC - Comparison with other PDF parsing tools
# MAGIC - Storing parsed content in Delta tables

# COMMAND ----------

from loguru import logger
from databricks.connect import DatabricksSession

from model_risk_agent.config import load_config, get_env
from model_risk_agent.data_processor import DataProcessor

# COMMAND ----------

spark = DatabricksSession.builder.getOrCreate()
logger.info("Using Databricks Connect Spark session")

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

# Initialize the DataProcessor (reusable class from arxiv_curator package)
processor = DataProcessor(spark=spark, config=cfg)

logger.info(f"Catalog: {cfg.catalog}, Schema: {cfg.schema}, Volume: {cfg.volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Download PDFs, Parse, and Create Chunks
# MAGIC
# MAGIC The `DataProcessor` class from `model_risk_agent.data_processor` handles the full pipeline:
# MAGIC 1. Download PDFs for unprocessed papers from regulatory and academic sources
# MAGIC 2. Update `mra_source_documents` table with processed timestamps and volume paths
# MAGIC 3. Parse PDFs using AI Parse Documents (`ai_parse_document`)
# MAGIC 4. Extract and clean text chunks, join with paper metadata
# MAGIC 5. Write chunks to `mra_chunks` table with Change Data Feed enabled
# MAGIC
# MAGIC The same class is used in `resources/deployment_scripts/process_data.py` for scheduled runs.

# COMMAND ----------

processor.process_and_save()

# COMMAND ----------
# MAGIC %sql
DESCRIBE TABLE llmops_dev.lennon_jcruz.mra_ai_parsed_docs;
# COMMAND ----------
# MAGIC %sql
SELECT path, processed, length(parsed_content) AS json_len
FROM llmops_dev.lennon_jcruz.mra_ai_parsed_docs
LIMIT 10;
# COMMAND ----------
# MAGIC %sql

SELECT count(*) AS n_parsed
FROM llmops_dev.lennon_jcruz.mra_ai_parsed_docs;

# COMMAND ----------
# MAGIC %sql
SELECT path, substr(parsed_content, 1, 500) AS preview
FROM llmops_dev.lennon_jcruz.mra_ai_parsed_docs
LIMIT 3;
# COMMAND ----------
# MAGIC %md
# MAGIC ## Inspect Chuncks
# COMMAND ----------
# MAGIC %sql
SELECT corpus,
       count(*) AS n_chunks,
       round(avg(length(text)), 1) AS avg_chars,
       min(length(text)) AS min_chars,
       max(length(text)) AS max_chars
FROM llmops_dev.lennon_jcruz.mra_chunks_table
GROUP BY corpus;

# COMMAND ----------
# MAGIC %sql
SELECT id, paper_id, corpus, length(text) AS chars, substr(text, 1, 200) AS preview
FROM llmops_dev.lennon_jcruz.mra_chunks_table
LIMIT 20;