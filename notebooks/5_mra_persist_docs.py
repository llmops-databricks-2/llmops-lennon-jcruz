# Databricks notebook source
#COMMAND ----------
%pip install --force-reinstall --no-deps /Volumes/llmops_dev/lennon_jcruz/packages/python/model_risk_agent-0.0.1-py3-none-any.whl
# COMMAND ----------
dbutils.library.restartPython()
# COMMAND ----------
from uuid import uuid4
from model_risk_agent.config import get_env, load_config
from model_risk_agent.model_doc_store import ModelDocStore

# COMMAND ----------
MODEL_DOC_PATH = "/Volumes/llmops_dev/lennon_jcruz/mra_model_docs/tests/model_deck.md"
MODEL_NAME = "Give Me Some Credit retail PD"

# COMMAND ----------
cfg = load_config(env=get_env())
# CREATE VOLUME IF NOT EXISTS {cfg.catalog}.{cfg.schema}.{cfg.model_doc_volume}
store = ModelDocStore(cfg)
run_id = str(uuid4())
out = store.persist_from_path(run_id, MODEL_DOC_PATH, model_name=MODEL_NAME)

# COMMAND ----------
dbutils.fs.ls("/Volumes/llmops_dev/lennon_jcruz/mra_model_docs/runs")