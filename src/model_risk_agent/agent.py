from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

from model_risk_agent.config import ProjectConfig
from model_risk_agent.mcp import ToolInfo
from model_risk_agent.model_doc_store import ModelDocStore, PersistedModelDoc
from model_risk_agent.prompts import (
    SYSTEM_PROMPT,
    build_assessment_user_message,
    parse_decision,
)
from model_risk_agent.tools import build_mra_tools
from model_risk_agent.vector_search import LakebasePgVectorManager


@dataclass(frozen=True)
class AssessmentResult:
    run_id: str
    model_name: str
    markdown_path: str
    report: str
    decision: str | None


class ModelRiskAgent:
    """A model risk agent that can call tools to assess a model."""
    
    def __init__(self, llm_endpoint: str, system_prompt: str, tools: list[ToolInfo], workspace: WorkspaceClient):
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self._tools_dict = {tool.name: tool for tool in tools}
        self.workspace = workspace
        self._client = OpenAI(
            api_key=self.workspace.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{self.workspace.config.host.rstrip('/')}/ai-gateway/mlflow/v1"
        )
    
    def get_tool_specs(self) -> list[dict]:
        """Get tool specifications for the LLM."""
        return [tool.spec for tool in self._tools_dict.values()]
    
    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool by name."""
        if tool_name not in self._tools_dict:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self._tools_dict[tool_name].exec_fn(**args)
    
    def chat(self, user_message: str, max_iterations: int = 6, max_tool_rounds: int = 3) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
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


def assess(
    config: ProjectConfig,
    markdown: str,
    *,
    model_name: str,
    run_id: str | None = None,
    question: str | None = None,
    workspace: WorkspaceClient | None = None,
    max_tool_rounds: int = 3,   # hard stop on tool use before text answer
    max_iterations: int = 6,
) -> AssessmentResult:
    run_id = run_id or str(uuid4())

    persisted: PersistedModelDoc = ModelDocStore(config).persist(
        run_id, markdown, model_name=model_name
    )

    w = workspace or WorkspaceClient()
    pgvector = LakebasePgVectorManager(config=config, workspace=w)
    tools = build_mra_tools(pgvector)

    agent = ModelRiskAgent(
        llm_endpoint=config.llm_endpoint,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        workspace=w,
    )

    user_message = build_assessment_user_message(
        run_id=persisted.run_id,
        model_name=persisted.model_name,
        volume_path=persisted.markdown_path,
        deck_md=markdown,
        question=question,
    )
    report = agent.chat(user_message, max_iterations=max_iterations, max_tool_rounds=max_tool_rounds)

    return AssessmentResult(
        run_id=persisted.run_id,
        model_name=persisted.model_name,
        markdown_path=persisted.markdown_path,
        report=report,
        decision=parse_decision(report),
    )   