# Databricks notebook source
# MAGIC %pip install --force-reinstall --no-deps  /Volumes/llmops_dev/lennon_jcruz/packages/python/model_risk_agent-0.0.1-py3-none-any.whl
# MAGIC %pip install "psycopg[binary]==3.3.2" "loguru==0.7.3" "openai==2.14.0" "databricks-mcp==0.4.0" "mcp>=1.13,<2"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pathlib import Path
from model_risk_agent.config import get_env, load_config
from model_risk_agent.agent import assess
from IPython.display import Markdown, display



MODEL_DOC_PATH = "/Volumes/llmops_dev/lennon_jcruz/mra_model_docs/tests/model_deck.md"
MODEL_NAME = "Give Me Some Credit retail PD"

cfg = load_config(env=get_env())
md = Path(MODEL_DOC_PATH).read_text()

result = assess(cfg, md, model_name=MODEL_NAME, max_tool_rounds=5)

display(Markdown(f"**Decision:** `{result.decision}`\n\n---\n\n{result.report}"))