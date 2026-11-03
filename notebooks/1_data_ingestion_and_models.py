# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install arxiv

# COMMAND ----------

# MAGIC %pip install --force-reinstall /Workspace/Users/lennon.jcruz@gmail.com/.bundle/dev/llmops-lennon-jcruz/files/notebooks/model_risk_agent-0.0.1-py3-none-any.whl

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession

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

# list regulatory sources

regulatory_sources = load_regulatory_sources("../regulatory_sources.yml")
print(regulatory_sources)

# COMMAND ----------

# Download and store public PDFs plus log on MRA table

data_processor = DataProcessor(spark, cfg)
data_processor.download_and_store_public_pdfs(regulatory_sources)

# COMMAND ----------

# Download and store academic PDFs plus log on MRA table

data_processor.download_and_store_papers()

# COMMAND ----------

# MAGIC %md
# MAGIC # Foundation Models

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

# COMMAND ----------

workspace_client = WorkspaceClient()
model_service_name = "course_ops.gateway.chat"

logger.info(f"Workspace: {workspace_client.config.host}")
logger.info(f"Unity Catalog model service: {model_service_name}")

# COMMAND ----------

for service in workspace_client.serving_endpoints.list():
    print(f"- {service.name}")

# COMMAND ----------

databricks_token = workspace_client.tokens.create(lifetime_seconds=1200).token_value

client = OpenAI(
    api_key=databricks_token,
    base_url=f"{workspace_client.config.host}/ai-gateway/mlflow/v1",
)

response = client.responses.create(
    model=model_service_name,
    max_output_tokens=256,
    input=[
        {"role": "user", "content": [{"type": "input_text", "text": "Hello!"}]},
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello! How can I assist you today?"}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "What is SR11-7?"}],
        },
    ],
)

logger.info("Response:")
logger.info(response.output_text)
if response.usage is not None:
    logger.info(f"Tokens used: {response.usage.total_tokens}")
    logger.info(f"Input tokens: {response.usage.input_tokens}")
    logger.info(f"Output tokens: {response.usage.output_tokens}")

# COMMAND ----------
