"""Prompts for the Model Risk Analyst agent."""

SYSTEM_PROMPT = """You are a US model-risk analyst for retail PD / credit-style models.

Rules:
- Use ONLY: search_regulatory, search_academic, hybrid_search_mra.
- Ground findings in (1) the model documentation and (2) tool evidence.
- Cite chunk id and title from tools. Do not invent regulations or metrics.
- Decision MUST be exactly one of: recommend | recommend with actions | deny
"""

REPORT_TEMPLATE = """# Model Risk Assessment
## Decision
<recommend | recommend with actions | deny>
## Findings
- ...
## Evidence
- [corpus] title (id=...) — short cite tied to a finding
## Actions
- ...
## Gaps / unknowns
- ...
"""

ALLOWED_DECISIONS = frozenset({"recommend", "recommend with actions", "deny"})


def build_assessment_user_message(
    *,
    run_id: str,
    model_name: str,
    volume_path: str,
    deck_md: str,
    question: str | None = None,
) -> str:
    """Build the user message that carries deck, metadata, and report template.

    Args:
        run_id: Assessment run identifier.
        model_name: Model name for the prompt header.
        volume_path: UC volume path of the persisted deck.
        deck_md: Model documentation Markdown body.
        question: Optional task override; default is a US retail PD assessment.

    Returns:
        Full user message string for the agent chat turn.

    """
    task = question or (
        "Assess this model for US retail PD use before production. "
        "Search regulatory guidance (and academic if needed) to support findings."
    )
    return f"""## Assessment metadata
- run_id: {run_id}
- model_name: {model_name}
- volume_path: {volume_path}

## Model documentation
{deck_md}

## Task
{task}
Return ONLY this Markdown structure:

{REPORT_TEMPLATE}
"""


def parse_decision(report: str) -> str | None:
    """Extract Decision label; tolerate **deny** bolding."""
    import re

    m = re.search(r"##\s*Decision\s*\n+\s*(.+?)(?:\n|$)", report or "", flags=re.I)
    if not m:
        return None
    cleaned = re.sub(r"[*_`#]", "", m.group(1)).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned in ALLOWED_DECISIONS:
        return cleaned
    for d in sorted(ALLOWED_DECISIONS, key=len, reverse=True):
        if cleaned.startswith(d):
            return d
    return None
