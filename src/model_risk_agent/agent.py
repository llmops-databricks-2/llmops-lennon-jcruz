from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import uuid4

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

from model_risk_agent.config import ProjectConfig
from model_risk_agent.mcp import ToolInfo
from model_risk_agent.memory import LakebaseMemory
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

    def __init__(
        self,
        llm_endpoint: str,
        system_prompt: str,
        tools: list[ToolInfo],
        workspace: WorkspaceClient,
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self._tools_dict = {tool.name: tool for tool in tools}
        self.workspace = workspace
        self._client = OpenAI(
            api_key=self.workspace.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{self.workspace.config.host.rstrip('/')}/ai-gateway/mlflow/v1",
        )
        logger.info(
            "ModelRiskAgent ready endpoint={} tools={}",
            llm_endpoint,
            list(self._tools_dict),
        )

    def get_tool_specs(self) -> list[dict]:
        """Get tool specifications for the LLM."""
        return [tool.spec for tool in self._tools_dict.values()]

    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool by name."""
        if tool_name not in self._tools_dict:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self._tools_dict[tool_name].exec_fn(**args)

    def chat(
        self,
        user_message: str,
        *,
        previous_messages: list[dict] | None = None,
        max_iterations: int = 6,
        max_tool_rounds: int = 3,
    ) -> str:
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        if previous_messages:
            messages.extend(previous_messages)
            logger.info("chat loaded {} prior message(s) from memory", len(previous_messages))
        messages.append({"role": "user", "content": user_message})
        logger.info(
            "chat start user_chars={} max_iterations={} max_tool_rounds={}",
            len(user_message),
            max_iterations,
            max_tool_rounds,
        )

        tool_rounds = 0
        for iteration in range(max_iterations):
            use_tools = tool_rounds < max_tool_rounds
            logger.debug(
                "chat iteration={} tool_rounds={} use_tools={}",
                iteration + 1,
                tool_rounds,
                use_tools,
            )
            response = self._client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=self.get_tool_specs() if use_tools else None,
            )
            assistant_message = response.choices[0].message

            if use_tools and assistant_message.tool_calls:
                tool_rounds += 1
                messages.append(
                    {
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
                    }
                )
                for tool_call in assistant_message.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    logger.info("Calling tool: {}({})", name, args)
                    if name not in self._tools_dict:
                        result = (
                            f"Error: unknown tool '{name}'. "
                            "Available: search_regulatory, search_academic, hybrid_search_mra. "
                            "Answer using search results already in the conversation."
                        )
                        logger.warning("unknown tool requested: {}", name)
                    else:
                        try:
                            result = self.execute_tool(name, args)
                            logger.info(
                                "tool {} returned {} chars",
                                name,
                                len(str(result)),
                            )
                        except Exception as e:
                            result = f"Error: {e}"
                            logger.exception("tool {} failed", name)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )
                if tool_rounds >= max_tool_rounds:
                    logger.info("tool round cap reached; forcing final text answer")
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Do not call more tools. Using only the tool results above, "
                                "write the final answer with citations (id + title)."
                            ),
                        }
                    )
                continue

            text = (assistant_message.content or "").strip()
            if text:
                logger.info("chat finished with text answer chars={}", len(text))
                return text
            logger.warning("empty assistant content on iteration {}; nudging", iteration + 1)
            messages.append(
                {
                    "role": "user",
                    "content": "Please provide the final written answer now with no tool calls.",
                }
            )

        logger.error("chat hit max iterations={}", max_iterations)
        return "Max iterations reached."


def assess(
    config: ProjectConfig,
    markdown: str,
    *,
    model_name: str,
    run_id: str | None = None,
    question: str | None = None,
    workspace: WorkspaceClient | None = None,
    max_tool_rounds: int = 3,
    max_iterations: int = 6,
    session_id: str | None = None,
) -> AssessmentResult:
    run_id = run_id or str(uuid4())
    w = workspace or WorkspaceClient()
    store = ModelDocStore(config)

    text = (markdown or "").strip()
    if text:
        persisted = store.persist(run_id, text, model_name=model_name)
        deck_md = text
        logger.info(
            "persisted model doc run_id={} path={}",
            persisted.run_id,
            persisted.markdown_path,
        )
    else:
        # Follow-up turn: reuse volume artifact; do not rewrite.
        if not run_id:
            raise ValueError("run_id is required when markdown is empty")
        deck_md = store.read_markdown(model_name, run_id)
        md_path = store.artifact_path(model_name, run_id)
        persisted = PersistedModelDoc(
            run_id=run_id,
            model_name=ModelDocStore._safe_model_name(model_name),
            markdown_path=str(md_path),
            runs_dir=str(store.runs_dir),
        )
        logger.info(
            "follow-up assess; reused model doc run_id={} path={}",
            persisted.run_id,
            persisted.markdown_path,
        )

    memory = None
    history: list[dict] = []
    if session_id:
        memory = LakebaseMemory(config.lakebase_project_id, workspace=w)
        memory.ensure_schema()
        history = memory.load_messages(session_id)
        logger.info(
            "session_id={} loaded {} message(s) from memory",
            session_id,
            len(history),
        )

    pgvector = LakebasePgVectorManager(config=config, workspace=w)
    tools = build_mra_tools(pgvector)
    logger.info("built {} search tool(s)", len(tools))

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
        deck_md=deck_md,
        question=question,
    )

    report = agent.chat(
        user_message,
        previous_messages=history,
        max_iterations=max_iterations,
        max_tool_rounds=max_tool_rounds,
    )
    decision = parse_decision(report)
    logger.info("assessment decision={} report_chars={}", decision, len(report or ""))

    if memory and session_id:
        memory.save_messages(
            session_id,
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": report},
            ],
        )
        logger.info("saved 2 message(s) to session_id={}", session_id)

    return AssessmentResult(
        run_id=persisted.run_id,
        model_name=persisted.model_name,
        markdown_path=persisted.markdown_path,
        report=report,
        decision=decision,
    )
