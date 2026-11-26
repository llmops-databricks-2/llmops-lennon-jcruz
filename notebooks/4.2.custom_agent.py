# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 4.2: Custom Agent with Tracing
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Integrating tracing into agents
# MAGIC - Tracing LLM calls
# MAGIC - Tracing tool executions
# MAGIC - Session and request tracking
# MAGIC - End-to-end agent tracing
# MAGIC - Performance analysis


# COMMAND ----------

from datetime import datetime
from uuid import uuid4
import random
import os
from dotenv import load_dotenv

import mlflow
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.types.responses import (
    ResponsesAgentRequest,
)
from pyspark.sql import SparkSession

from arxiv_curator.config import load_config, get_env
from arxiv_curator.agent import ArxivAgent


# COMMAND ----------

# Setup MLflow tracking
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

# Set experiment
mlflow.set_experiment(cfg.experiment_name)

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Agent with Tracing - Architecture
# MAGIC
# MAGIC ```
# MAGIC User Request
# MAGIC     ↓
# MAGIC ┌──────────────────────────────────────────────┐
# MAGIC │  @mlflow.trace(AGENT)                        │
# MAGIC │  predict()                                    │
# MAGIC │    ├─ Update trace metadata                   │
# MAGIC │    │  (session_id, request_id, git_sha)       │
# MAGIC │    │                                           │
# MAGIC │    ├─ @mlflow.trace(RETRIEVER)                │
# MAGIC │    │  load_memory(session_id)                  │
# MAGIC │    │  → prepend past messages to conversation  │
# MAGIC │    │                                           │
# MAGIC │    ├─ call_and_run_tools()                    │
# MAGIC │    │    ├─ @mlflow.trace(LLM)                 │
# MAGIC │    │    │  call_llm()                          │
# MAGIC │    │    │                                      │
# MAGIC │    │    ├─ @mlflow.trace(TOOL)                │
# MAGIC │    │    │  execute_tool()                      │
# MAGIC │    │    │                                      │
# MAGIC │    │    └─ Loop until done                     │
# MAGIC │    │                                           │
# MAGIC │    └─ @mlflow.trace(CHAIN)                    │
# MAGIC │       save_memory(session_id, new_messages)    │
# MAGIC └──────────────────────────────────────────────┘
# MAGIC     ↓
# MAGIC Response + Complete Trace
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. ArxivAgent with Tracing
# MAGIC
# MAGIC The `ArxivAgent` class from `arxiv_curator.agent` provides:
# MAGIC - Full MLflow tracing integration
# MAGIC - Lakebase pgvector and Genie tool support
# MAGIC - Session and request tracking
# MAGIC - Automatic deployment metadata

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agent Architecture (from agent.py):
# MAGIC
# MAGIC ```python
# MAGIC class ArxivAgent(ResponsesAgent):
# MAGIC     def __init__(self, llm_endpoint, system_prompt, catalog, schema, genie_space_id):
# MAGIC         # Creates a Lakebase pgvector tool and optional Genie MCP tools
# MAGIC         ...
# MAGIC     
# MAGIC     @mlflow.trace(span_type=SpanType.TOOL)
# MAGIC     def execute_tool(self, tool_name, args):
# MAGIC         # Traced tool execution
# MAGIC         ...
# MAGIC     
# MAGIC     @mlflow.trace(span_type=SpanType.LLM)
# MAGIC     def call_llm(self, messages):
# MAGIC         # Traced LLM calls
# MAGIC         ...
# MAGIC     
# MAGIC     @mlflow.trace(span_type=SpanType.CHAIN)
# MAGIC     def call_and_run_tools(self, messages, ...):
# MAGIC         # Traced agentic loop
# MAGIC         ...
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create ArxivAgent Instance
# MAGIC
# MAGIC The agent automatically gets MCP tools for:
# MAGIC - Lakebase pgvector (search arXiv papers)
# MAGIC - Genie (query data with natural language)

# COMMAND ----------

# Create agent with MCP tools
agent = ArxivAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt="You are a helpful research assistant. Use vector search to find papers and Genie to query data.",
    catalog=cfg.catalog,
    schema=cfg.schema,
    genie_space_id=cfg.genie_space_id,
    lakebase_project_id=cfg.lakebase_project_id,
    embedding_endpoint=cfg.embedding_endpoint,
)

logger.info("✓ ArxivAgent created with MCP tools:")
logger.info(f"  - Lakebase pgvector project: {cfg.lakebase_project_id}")
logger.info(f"  - Genie Space: {cfg.genie_space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test the Traced Agent

# COMMAND ----------

# Generate trace identifiers
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

# Create request
test_request = ResponsesAgentRequest(
    input=[{
        "role": "user",
        "content": "Find papers about transformers and attention mechanisms"
    }],
    custom_inputs={
        "session_id": session_id,
        "request_id": request_id
    }
)

logger.info(f"Session ID: {session_id}")
logger.info(f"Request ID: {request_id}")
logger.info("Agent Response:")
logger.info("=" * 80)

# Call agent
response = agent.predict(test_request)
logger.info(response.output[-1].content)

logger.info("✓ Trace created! Check MLflow UI for complete trace.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Multi-Turn Conversation with Tracing

# COMMAND ----------

# Start a conversation
conversation_session = f"s-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{random.randint(100000, 999999)}"

logger.info(f"Conversation Session: {conversation_session}")

# Turn 1
request1 = ResponsesAgentRequest(
    input=[{
        "role": "user",
        "content": "Calculate 100 plus 50"
    }],
    custom_inputs={
        "session_id": conversation_session,
        "request_id": f"req-1-{uuid4().hex[:8]}"
    }
)

response1 = agent.predict(request1)
logger.info(f"Turn 1:")
logger.info(f"User: Calculate 100 plus 50")
logger.info(f"Agent: {response1.output[-1].content}")

# Turn 2 (with context)
request2 = ResponsesAgentRequest(
    input=[
        {"role": "user", "content": "Calculate 100 plus 50"},
        {"role": "assistant", "content": response1.output[-1].content},
        {"role": "user", "content": "Now multiply that by 2"}
    ],
    custom_inputs={
        "session_id": conversation_session,
        "request_id": f"req-2-{uuid4().hex[:8]}"
    }
)

response2 = agent.predict(request2)
logger.info(f"Turn 2:")
logger.info(f"User: Now multiply that by 2")
logger.info(f"Agent: {response2.output[-1].content}")

logger.info(f"✓ Multi-turn conversation traced with session: {conversation_session}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Analyzing Agent Traces

# COMMAND ----------

# Search traces by session (returns DataFrame)
session_traces_df = mlflow.search_traces(
    filter_string=f"request_metadata.`mlflow.trace.session` = '{conversation_session}'",
    order_by=["timestamp_ms ASC"]
)

logger.info(f"Traces for session {conversation_session}:")
logger.info("=" * 80)

if len(session_traces_df) > 0:
    logger.info(f"Available columns: {list(session_traces_df.columns)}")
    
    # Select only scalar columns to avoid Arrow conversion errors
    simple_cols = []
    for col in session_traces_df.columns:
        if col not in ['request', 'response', 'spans', 'inputs', 'outputs']:
            simple_cols.append(col)
    
    if simple_cols:
        display(session_traces_df[simple_cols].head())
    else:
        logger.info(str(session_traces_df.info()))
else:
    logger.info("No traces found for this session.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Performance Analysis

# COMMAND ----------

# Get recent agent traces (returns DataFrame)
recent_traces_df = mlflow.search_traces(
    order_by=["timestamp_ms DESC"],
    max_results=20
)

if len(recent_traces_df) > 0:
    logger.info("Performance Statistics:")
    logger.info("=" * 80)
    logger.info(f"Total traces: {len(recent_traces_df)}")
    
    # Calculate statistics if execution_time_ms column exists
    if 'execution_time_ms' in recent_traces_df.columns:
        durations = recent_traces_df['execution_time_ms'].dropna()
        if len(durations) > 0:
            logger.info(f"Avg duration: {durations.mean():.2f}ms")
            logger.info(f"Min duration: {durations.min():.2f}ms")
            logger.info(f"Max duration: {durations.max():.2f}ms")
    
    # Count by status if column exists
    if 'status' in recent_traces_df.columns:
        logger.info("By Status:")
        status_counts = recent_traces_df['status'].value_counts()
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count}")
    
    # Show sample of traces (select only simple columns to avoid Arrow conversion issues)
    logger.info("Sample Traces:")
    logger.info(f"Available columns: {list(recent_traces_df.columns)}")
    
    # Select only scalar columns for display
    simple_cols = []
    for col in recent_traces_df.columns:
        # Skip complex object columns that cause Arrow conversion errors
        if col not in ['request', 'response', 'spans', 'inputs', 'outputs']:
            simple_cols.append(col)
    
    if simple_cols:
        display(recent_traces_df[simple_cols].head())
    else:
        # Fallback: just show the info
        logger.info(str(recent_traces_df.info()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Trace Inspection

# COMMAND ----------

if len(recent_traces_df) > 0:
    # Get the most recent trace (first row)
    trace = recent_traces_df.iloc[0]
    
    print("Detailed Trace Inspection:")
    print("=" * 80)
    print(f"Request ID: {trace.get('request_id', 'N/A')}")
    print(f"Trace ID: {trace.get('trace_id', 'N/A')}")
    print(f"Duration: {trace.get('execution_time_ms', 'N/A')}ms")
    print(f"Status: {trace.get('status', 'N/A')}")
    
    # Tags
    if 'tags' in trace and trace['tags']:
        print(f"\nTags:")
        for key, value in trace['tags'].items():
            print(f"  {key}: {value}")
    
    # Metadata
    if 'request_metadata' in trace and trace['request_metadata']:
        print(f"\nMetadata:")
        for key, value in trace['request_metadata'].items():
            print(f"  {key}: {value}")
    
    # Spans
    if 'spans' in trace:
        spans_count = len(trace['spans']) if trace['spans'] else 0
        print(f"\nSpans: {spans_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Best Practices Summary

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tracing Best Practices:
# MAGIC
# MAGIC 1. **Always add session_id** for conversation tracking
# MAGIC 2. **Use unique request_id** for each request
# MAGIC 3. **Include git_sha** for version tracking
# MAGIC 4. **Trace all LLM calls** with SpanType.LLM
# MAGIC 5. **Trace all tool executions** with SpanType.TOOL
# MAGIC 6. **Use SpanType.CHAIN** for multi-step operations
# MAGIC 7. **Add deployment metadata** (endpoint, version)
# MAGIC 8. **Set span attributes** for debugging
# MAGIC 9. **Handle errors gracefully** in traces
# MAGIC 10. **Search traces** for analysis and debugging
