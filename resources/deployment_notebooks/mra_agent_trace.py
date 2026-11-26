# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Run ModelRiskAgent.assess once and confirm MLflow traces landed."""
# COMMAND ----------
# MAGIC %md
# MAGIC # MRA agent + tracing smoke
# MAGIC
# MAGIC Job notebook: loads `project_config.yml`, runs `assess()` on a model deck from the
# MAGIC model-doc volume, then searches the experiment for the new AGENT/LLM/TOOL spans.

# COMMAND ----------

import os
from pathlib import Path
from uuid import uuid4

import mlflow
from databricks.sdk.runtime import dbutils
from loguru import logger

from model_risk_agent.agent import assess
from model_risk_agent.config import load_config

dbutils.widgets.text("env", "dev")
dbutils.widgets.text(
    "model_doc_path",
    "/Volumes/llmops_dev/lennon_jcruz/mra_model_docs/tests/model_deck.md",
)
dbutils.widgets.text("model_name", "Give Me Some Credit retail PD")
dbutils.widgets.text("max_tool_rounds", "5")

env = dbutils.widgets.get("env")
model_doc_path = dbutils.widgets.get("model_doc_path")
model_name = dbutils.widgets.get("model_name")
max_tool_rounds = int(dbutils.widgets.get("max_tool_rounds"))

cfg = load_config("../../project_config.yml", env=env)
# UC-backed traces need a warehouse (lecture 4.0 pattern).
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = cfg.warehouse_id
logger.info(
    "config loaded schema={} experiment={} warehouse={} deck={}",
    cfg.full_schema_name,
    cfg.experiment_path,
    cfg.warehouse_id,
    model_doc_path,
)

# COMMAND ----------

mlflow.set_experiment(cfg.experiment_path)
session_id = f"job-trace-{uuid4().hex[:10]}"
markdown = Path(model_doc_path).read_text(encoding="utf-8")

result = assess(
    cfg,
    markdown,
    model_name=model_name,
    max_tool_rounds=max_tool_rounds,
    session_id=session_id,
)

logger.info("decision={}", result.decision)
logger.info("run_id={}", result.run_id)
logger.info("markdown_path={}", result.markdown_path)
logger.info("report_chars={}", len(result.report or ""))
logger.info("report_preview=\n{}", (result.report or "")[:2000])

# COMMAND ----------

experiment = mlflow.get_experiment_by_name(cfg.experiment_path)
traces_df = mlflow.search_traces(
    experiment_ids=[experiment.experiment_id],
    order_by=["timestamp_ms DESC"],
    max_results=5,
    sql_warehouse_id=cfg.warehouse_id,
)
logger.info("recent_traces={}", len(traces_df))
if len(traces_df) == 0:
    raise RuntimeError(f"No traces found in experiment {cfg.experiment_path}")

latest = traces_df.iloc[0]
logger.info(
    "latest_trace id={} status={} tags={}",
    latest.get("trace_id", latest.get("request_id", "n/a")),
    latest.get("status", "n/a"),
    latest.get("tags", {}),
)

dbutils.notebook.exit(
    f"decision={result.decision}; run_id={result.run_id}; "
    f"traces={len(traces_df)}; session_id={session_id}"
)
