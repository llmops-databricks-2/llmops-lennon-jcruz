"""Custom search tools for the Model Risk Analyst agent."""

import json

import mlflow
from mlflow.entities import Document, SpanType

from model_risk_agent.mcp import ToolInfo
from model_risk_agent.vector_search import LakebasePgVectorManager

SEARCH_REGULATORY_SPEC = {
    "type": "function",
    "function": {
        "name": "search_regulatory",
        "description": (
            "Search US model-risk regulatory guidance chunks only "
            "(corpus=regulatory). Use for SR 11-7 themes, documentation, "
            "monitoring, validation expectations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
}

SEARCH_ACADEMIC_SPEC = {
    "type": "function",
    "function": {
        "name": "search_academic",
        "description": ("Search academic papers only (corpus=academic). Use for academic research."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
}

HYBRID_SEARCH_MRA_SPEC = {
    "type": "function",
    "function": {
        "name": "hybrid_search_mra",
        "description": (
            "Hybrid vector + keyword search over MRA chunks. "
            "Omit corpus to search both regulatory and academic; "
            "or set corpus to 'regulatory' or 'academic'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "default": 5},
                "corpus": {
                    "type": "string",
                    "description": "Optional corpus filter: regulatory or academic",
                },
            },
            "required": ["query"],
        },
    },
}


def _format_hits(rows: list[dict]) -> str:
    hits = []
    for row in rows:
        hits.append(
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "corpus": row.get("corpus"),
                "score": round(float(row.get("score", 0.0)), 4),
                "excerpt": (row.get("text") or "")[:300],
            }
        )
    return json.dumps(hits, indent=2)


def _rows_as_documents(rows: list[dict]) -> list[Document]:
    """Convert pgvector rows into MLflow retriever Documents for tracing/judges."""
    docs: list[Document] = []
    for row in rows:
        docs.append(
            Document(
                id=str(row.get("id") or ""),
                page_content=(row.get("text") or "")[:1000],
                metadata={
                    "title": row.get("title"),
                    "corpus": row.get("corpus"),
                    "score": float(row.get("score") or 0.0),
                    "paper_id": row.get("paper_id"),
                },
            )
        )
    return docs


def _traced_search(span_name: str, rows: list[dict]) -> str:
    """Emit a RETRIEVER span (for RetrievalGroundedness) and return tool JSON."""
    with mlflow.start_span(name=span_name, span_type=SpanType.RETRIEVER) as span:
        span.set_inputs({"hit_count": len(rows)})
        span.set_outputs(_rows_as_documents(rows))
        return _format_hits(rows)


def build_mra_tools(pgvector: LakebasePgVectorManager) -> list[ToolInfo]:
    """Build regulatory, academic, and hybrid search tools bound to ``pgvector``.

    Args:
        pgvector: Lakebase pgvector manager used by the tool closures.

    Returns:
        List of ``ToolInfo`` entries ready for the agent registry.

    """

    def search_regulatory(query: str, num_results: int = 5) -> str:
        rows = pgvector.search(query=query, num_results=num_results, corpus="regulatory")
        return _traced_search("search_regulatory", rows)

    def search_academic(query: str, num_results: int = 5) -> str:
        rows = pgvector.search(query=query, num_results=num_results, corpus="academic")
        return _traced_search("search_academic", rows)

    def hybrid_search_mra(query: str, num_results: int = 5, corpus: str | None = None) -> str:
        rows = pgvector.hybrid_search(query=query, num_results=num_results, corpus=corpus)
        return _traced_search("hybrid_search_mra", rows)

    return [
        ToolInfo(name="search_regulatory", spec=SEARCH_REGULATORY_SPEC, exec_fn=search_regulatory),
        ToolInfo(name="search_academic", spec=SEARCH_ACADEMIC_SPEC, exec_fn=search_academic),
        ToolInfo(name="hybrid_search_mra", spec=HYBRID_SEARCH_MRA_SPEC, exec_fn=hybrid_search_mra),
    ]
