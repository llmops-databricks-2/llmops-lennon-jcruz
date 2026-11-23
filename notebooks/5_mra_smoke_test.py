# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "loguru",
#   "psycopg",
#   "openai==2.14.0",
# ]
# ///
# MAGIC %pip install --force-reinstall --no-deps  /Volumes/llmops_dev/lennon_jcruz/packages/python/model_risk_agent-0.0.1-py3-none-any.whl
# MAGIC %pip install "psycopg[binary]==3.3.2" "loguru==0.7.3" "openai==2.14.0" "databricks-mcp==0.4.0" "mcp>=1.13,<2"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient

from loguru import logger
from uuid import uuid4
from model_risk_agent.config import get_env, load_config
from model_risk_agent.model_doc_store import ModelDocStore
from model_risk_agent.vector_search import LakebasePgVectorManager
from model_risk_agent.tools import build_mra_tools
from model_risk_agent.mcp import ToolInfo

from openai import OpenAI

import json

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

# COMMAND ----------

env = get_env()
cfg = load_config("../project_config.yml", env)
w = WorkspaceClient()

pgvector = LakebasePgVectorManager(config=cfg, workspace=w)
tools = build_mra_tools(pgvector)
by_name = {t.name: t for t in tools}

logger.info(f"tools: {list(by_name)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call each tool

# COMMAND ----------

reg = by_name["search_regulatory"].exec_fn(
    query="model documentation and ongoing monitoring expectations",
    num_results=3,
)
print("=== search_regulatory ===")
print(reg)

acad = by_name["search_academic"].exec_fn(
    query="credit risk model validation",
    num_results=3,
)
print("=== search_academic ===")
print(acad)

hybrid = by_name["hybrid_search_mra"].exec_fn(
    query="SR 11-7",
    num_results=3,
    # corpus omitted → both
)
print("=== hybrid_search_mra ===")
print(hybrid)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simple Agent

# COMMAND ----------

class SimpleAgent:
    """A simple agent that can call tools in a loop."""
    
    def __init__(self, llm_endpoint: str, system_prompt: str, tools: list[ToolInfo]):
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self._tools_dict = {tool.name: tool for tool in tools}
        self._client = OpenAI(
            api_key=w.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{w.config.host.rstrip('/')}/ai-gateway/mlflow/v1"
        )
    
    def get_tool_specs(self) -> list[dict]:
        """Get tool specifications for the LLM."""
        return [tool.spec for tool in self._tools_dict.values()]
    
    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool by name."""
        if tool_name not in self._tools_dict:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self._tools_dict[tool_name].exec_fn(**args)
    
    def chat(self, user_message: str, max_iterations: int = 6) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        max_tool_rounds = 3  # hard stop on tool use
        tool_rounds = 0

        for _ in range(max_iterations):
            use_tools = tool_rounds < max_tool_rounds
            response = self._client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=self.get_tool_specs() if use_tools else None,
            )
            assistant_message = response.choices[0].message

            if use_tools and assistant_message.tool_calls:
                tool_rounds += 1
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_message.tool_calls
                    ],
                })
                for tool_call in assistant_message.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    logger.info(f"Calling tool: {name}({args})")
                    if name not in self._tools_dict:
                        result = (
                            f"Error: unknown tool '{name}'. "
                            "Available: search_regulatory, search_academic, hybrid_search_mra. "
                            "Answer using search results already in the conversation."
                        )
                    else:
                        try:
                            result = self.execute_tool(name, args)
                        except Exception as e:
                            result = f"Error: {e}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    })
                # After last allowed tool round, force a text answer next
                if tool_rounds >= max_tool_rounds:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Do not call more tools. Using only the tool results above, "
                            "write the final answer with citations (id + title)."
                        ),
                    })
                continue

            # Final text turn
            text = (assistant_message.content or "").strip()
            if text:
                return text
            messages.append({
                "role": "user",
                "content": "Please provide the final written answer now with no tool calls.",
            })

        return "Max iterations reached."

# COMMAND ----------

SYSTEM = """You are a US model-risk analyst assistant.

Rules:
- You may use ONLY these tools: search_regulatory, search_academic, hybrid_search_mra.
- Do NOT call any other tool names (no open_file, no browse, no fetch).
- Call at most TWO searches total, then answer from those results.
- If results are thin, say what is missing; do not keep searching.
- Cite chunk id and title from tool JSON. Do not invent guidance.
"""

agent = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=SYSTEM,
    tools=tools,  # from build_mra_tools
)

# COMMAND ----------

answer = agent.chat(
    "What does US model-risk guidance expect for ongoing monitoring of a retail PD model? "
    "Use at most two tool calls, then give a final answer.",
    max_iterations=6,
)
print(repr(answer))
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wire Model Deck

# COMMAND ----------

SYSTEM = """You are a US model-risk analyst for retail PD / credit-style models.

Rules:
- Use ONLY: search_regulatory, search_academic, hybrid_search_mra.
- Do NOT call any other tool (no open_file).
- Call at most TWO searches, then write the assessment.
- Ground findings in (1) the model documentation in the user message and (2) tool evidence.
- Cite chunk id and title from tools. Do not invent regulations or metrics.
- Decision MUST be exactly one of: recommend | recommend with actions | deny
"""

agent = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=SYSTEM,
    tools=tools,
)

# COMMAND ----------

from pathlib import Path
import re

# `out` came from store.persist_from_path(...)
deck_md = Path(out.markdown_path).read_text(encoding="utf-8")
# or: deck_md = store.read_markdown(out.model_name, out.run_id)

user_message = f"""## Assessment metadata
- run_id: {out.run_id}
- model_name: {out.model_name}
- volume_path: {out.markdown_path}

## Model documentation
{deck_md}

## Task
Assess this model for US retail PD use before production.
Search regulatory guidance (and academic if needed) to support findings.
Return ONLY this Markdown structure:

# Model Risk Assessment
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

print(f"deck chars: {len(deck_md)}; run_id={out.run_id}")

# COMMAND ----------

ALLOWED = {"recommend", "recommend with actions", "deny"}

report = agent.chat(user_message, max_iterations=6)
print(report)

m = re.search(
    r"##\s*Decision\s*\n+\s*(.+?)(?:\n|$)",
    report or "",
    flags=re.I,
)

cleaned = re.sub(r"[*_`#]", "", raw).strip().lower()
cleaned = re.sub(r"\s+", " ", cleaned)

decision = next((d for d in ALLOWED if cleaned == d or cleaned.startswith(d)), None)
print("raw:", repr(raw), "→", decision)
assert decision in ALLOWED, f"Decision missing or invalid: {raw!r}"

# COMMAND ----------

from pathlib import Path
from model_risk_agent.config import get_env, load_config
from model_risk_agent.agent import assess

cfg = load_config("../project_config.yml", get_env())
cfg