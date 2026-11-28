# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Run ModelRiskAgent evaluation over eval/cases.yaml and log scorer metrics."""
# COMMAND ----------
# MAGIC %md
# MAGIC # MRA agent evaluation
# MAGIC
# MAGIC Job notebook: loads config, sets the UC tracing warehouse, runs
# MAGIC ``evaluate_agent`` on the reviewed ``eval/cases.yaml`` dataset.

# COMMAND ----------

import os

import mlflow
from databricks.sdk.runtime import dbutils
from loguru import logger

from model_risk_agent.config import load_config
from model_risk_agent.evaluation import evaluate_agent, load_eval_cases

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("cases_path", "../../eval/cases.yaml")
dbutils.widgets.text("max_tool_rounds", "3")

env = dbutils.widgets.get("env")
cases_path = dbutils.widgets.get("cases_path")
max_tool_rounds = int(dbutils.widgets.get("max_tool_rounds"))

cfg = load_config("../../project_config.yml", env=env)
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = cfg.warehouse_id
mlflow.set_experiment(cfg.experiment_path)

cases = load_eval_cases(cases_path)
logger.info(
    "eval start schema={} experiment={} cases={} path={}",
    cfg.full_schema_name,
    cfg.experiment_path,
    len(cases),
    cases_path,
)

# COMMAND ----------

results = evaluate_agent(
    cfg,
    cases_path=cases_path,
    max_tool_rounds=max_tool_rounds,
)

metrics = getattr(results, "metrics", None) or {}
run_id = getattr(results, "run_id", None)
logger.info("eval_run_id={}", run_id)
logger.info("eval_metrics={}", metrics)

tables = getattr(results, "tables", None) or {}
eval_table = tables.get("eval_results")
if eval_table is not None:
    logger.info("eval_results_rows={}", len(eval_table))
    logger.info("eval_results_columns={}", list(eval_table.columns))
    # Keep logs readable: one compact row preview per case when available.
    preview_cols = [c for c in eval_table.columns if c in {"inputs", "outputs"} or "score" in c or "assessment" in c]
    if not preview_cols:
        preview_cols = list(eval_table.columns)[:8]
    logger.info("eval_results_preview=\n{}", eval_table[preview_cols].to_string())

metric_summary = ", ".join(f"{k}={v}" for k, v in sorted(metrics.items())[:20])
dbutils.notebook.exit(f"cases={len(cases)}; eval_run_id={run_id}; metrics={{{metric_summary}}}")
