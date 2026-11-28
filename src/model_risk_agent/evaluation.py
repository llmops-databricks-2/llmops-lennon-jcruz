"""MLflow GenAI scorers and evaluate runner for the Model Risk Analyst agent.

Course pattern (lecture ``arxiv_curator.evaluation``): Guidelines for policy
checks, ``@mlflow.genai.scorer`` for deterministic code checks, plus
``load_eval_cases`` / ``evaluate_agent`` over ``eval/cases.yaml``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow.genai.scorers import Guidelines, RetrievalGroundedness

from model_risk_agent.agent import assess
from model_risk_agent.config import ProjectConfig, _resolve_path
from model_risk_agent.prompts import ALLOWED_DECISIONS, parse_decision

# Guidelines call the Databricks OpenAI-compatible /serving-endpoints surface.
# Unity Gateway names like course_ops.gateway.chat 404 there (ENDPOINT_NOT_FOUND).
# Use a foundation serving-endpoint id under databricks:/ (not endpoints:/ — that
# path hits deployments.get_endpoint and fails in this workspace).
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "databricks:/databricks-claude-haiku-4-5")

# Domain Guidelines (LLM-backed binary pass/fail) — Assignment 4 "domain expectations".
decision_policy_guideline = Guidelines(
    name="decision_policy",
    guidelines=[
        "The response must state exactly one Decision label: "
        "recommend, recommend with actions, or deny (no other labels).",
        "If material risk gaps remain but the model is not clearly unsafe, prefer "
        "'recommend with actions' over an unqualified 'recommend'.",
        "Use 'deny' only when evidence shows the model is unfit for production "
        "or documentation is too weak to support approval.",
    ],
    model=JUDGE_MODEL,
)

# Guidelines only see inputs+outputs (not TOOL payloads). Accept explicit
# [corpus] title (id=...) Evidence lines as tool-grounded cites.
grounded_evidence_guideline = Guidelines(
    name="grounded_evidence",
    guidelines=[
        "Findings must be grounded in the model documentation and/or Evidence cites.",
        "Do not invent regulation titles, paper titles, or metrics that are not in "
        "the model documentation or the Evidence section.",
        "When regulatory or academic claims are made, Evidence bullets must use "
        "the form [corpus] title (id=chunk_id). Treat that form as a valid tool cite; "
        "do not fail solely because raw retrieval payloads are absent from the request.",
    ],
    model=JUDGE_MODEL,
)

report_sections_guideline = Guidelines(
    name="report_sections",
    guidelines=[
        "The response must follow the Model Risk Assessment Markdown structure "
        "with Decision, Findings, Evidence, Actions, and Gaps / unknowns sections.",
        "Actions must be concrete and actionable when the decision is 'recommend with actions' or 'deny'.",
    ],
    model=JUDGE_MODEL,
)

# Trace-aware groundedness against the last RETRIEVER span (search tools emit these).
retrieval_groundedness = RetrievalGroundedness(
    name="retrieval_groundedness",
    model=JUDGE_MODEL,
)


def _output_text(outputs: object) -> str:
    """Normalize MLflow scorer ``outputs`` to a plain string."""
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, dict) and "text" in first:
            return str(first["text"])
        if isinstance(first, str):
            return first
        return str(first)
    if isinstance(outputs, dict) and "text" in outputs:
        return str(outputs["text"])
    if isinstance(outputs, str):
        return outputs
    return str(outputs)


@mlflow.genai.scorer
def decision_label_check(outputs: object) -> bool:
    """Code check: report contains a parseable allowed Decision label.

    Assignment 4 "code check" — no LLM; uses ``parse_decision``.
    """
    return parse_decision(_output_text(outputs)) in ALLOWED_DECISIONS


@mlflow.genai.scorer
def report_structure_check(outputs: object) -> bool:
    """Code check: required Markdown section headings are present."""
    text = _output_text(outputs)
    required = (
        r"##\s*Decision\b",
        r"##\s*Findings\b",
        r"##\s*Evidence\b",
        r"##\s*Actions\b",
        r"##\s*Gaps",
    )
    return all(re.search(pat, text, flags=re.I) for pat in required)


DEFAULT_SCORERS = [
    decision_label_check,
    report_structure_check,
    decision_policy_guideline,
    grounded_evidence_guideline,
    report_sections_guideline,
    retrieval_groundedness,
]


def load_eval_cases(cases_path: str | Path = "eval/cases.yaml") -> list[dict[str, Any]]:
    """Load reviewed eval cases into ``mlflow.genai.evaluate`` row shape.

    Each YAML case becomes::

        {
            "inputs": {case_id, model_name, question, markdown},
            "expectations": {"expected_decision": ...},
        }

    Args:
        cases_path: Path to ``cases.yaml`` (cwd / parents resolved).

    Returns:
        List of eval rows ready for ``mlflow.genai.evaluate``.

    """
    path = _resolve_path(str(cases_path))
    with path.open(encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    cases = payload.get("cases") or []
    if not cases:
        raise ValueError(f"No cases found in {path}")

    rows: list[dict[str, Any]] = []
    for raw in cases:
        case_id = str(raw["case_id"])
        expected = str(raw["expected_decision"]).strip().lower()
        if expected not in ALLOWED_DECISIONS:
            raise ValueError(
                f"case {case_id}: expected_decision must be one of {sorted(ALLOWED_DECISIONS)}, got {expected!r}"
            )
        rows.append(
            {
                "inputs": {
                    "case_id": case_id,
                    "model_name": str(raw["model_name"]),
                    "question": str(raw.get("question") or "").strip() or None,
                    "markdown": str(raw["markdown"]),
                },
                "expectations": {"expected_decision": expected},
            }
        )
    return rows


def evaluate_agent(
    cfg: ProjectConfig,
    cases_path: str | Path = "eval/cases.yaml",
    *,
    scorers: list | None = None,
    max_tool_rounds: int = 3,
    max_iterations: int = 6,
) -> object:
    """Run ``assess`` over the reviewed dataset with DEFAULT_SCORERS.

    Course-shaped entrypoint: build ``predict_fn`` from inputs keys, then
    ``mlflow.genai.evaluate``. Caller should ``mlflow.set_experiment`` and set
    ``MLFLOW_TRACING_SQL_WAREHOUSE_ID`` first (notebook / job).

    Args:
        cfg: Active environment config.
        cases_path: Reviewed YAML dataset path.
        scorers: Override scorers; default ``DEFAULT_SCORERS``.
        max_tool_rounds: Cap tool rounds inside each ``assess``.
        max_iterations: Cap LLM iterations inside each ``assess``.

    Returns:
        MLflow GenAI evaluation result object.

    """
    eval_data = load_eval_cases(cases_path)
    active_scorers = scorers if scorers is not None else list(DEFAULT_SCORERS)

    def predict_fn(
        markdown: str,
        model_name: str,
        question: str | None = None,
        case_id: str | None = None,
    ) -> str:
        """Invoke the traced assess entrypoint for one eval row."""
        result = assess(
            cfg,
            markdown,
            model_name=model_name,
            question=question,
            max_tool_rounds=max_tool_rounds,
            max_iterations=max_iterations,
            session_id=f"eval-{case_id}" if case_id else None,
        )
        return result.report

    return mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=eval_data,
        scorers=active_scorers,
    )
