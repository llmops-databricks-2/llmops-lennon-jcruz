# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Lecture 2.5: Synced Tables and pgvector on Lakebase
# MAGIC
# MAGIC The chunks table already holds an `embedding` column (written by
# MAGIC `process_chunks` and backfilled in Lecture 2.4). A Lakebase Synced Table
# MAGIC copies it into Postgres, and we search it natively with pgvector.
# MAGIC
# MAGIC ```text
# MAGIC mra_chunks_table (Unity Catalog Delta: text + embedding)
# MAGIC          │  Synced Table (managed, read-only copy)
# MAGIC          ▼
# MAGIC Lakebase Postgres: mra_chunks_synced
# MAGIC          ├── HNSW cosine index on the embedding
# MAGIC          ├── metadata filters
# MAGIC          └── PostgreSQL full-text ranking
# MAGIC ```
# MAGIC
# MAGIC No embeddings are generated or inserted here: Databricks owns the copy, we
# MAGIC own the indexes and the queries.

# COMMAND ----------

# MAGIC %pip install --force-reinstall /Volumes/llmops_dev/lennon_jcruz/packages/python/model_risk_agent-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient

from model_risk_agent.config import get_env, load_config
from model_risk_agent.vector_search import LakebasePgVectorManager

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env()
cfg = load_config("../project_config.yml", env)
w = WorkspaceClient()

source_table = f"{cfg.catalog}.{cfg.schema}.mra_chunks_table"
synced_table = f"{cfg.catalog}.{cfg.schema}.mra_chunks_synced"
postgres_branch = f"projects/{cfg.lakebase_project_id}/branches/production"

# Keep creation disabled until the source table and permissions have been checked.
RUN_CREATE_SYNC = True
# Start a sync of the existing Synced Table from code (see section 3)
RUN_SYNC_NOW = True

logger.info(f"Environment: {env}")
logger.info(f"Source table: {source_table}")
logger.info(f"Synced table: {synced_table}")
logger.info(f"Lakebase project: {cfg.lakebase_project_id}")
logger.info(f"Embedding endpoint: {cfg.embedding_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Validate the Delta source
# MAGIC
# MAGIC Synced Tables require a stable, non-null primary key. The chunk pipeline
# MAGIC creates `id` for every chunk, so it is used as the synchronization key.
# MAGIC Triggered and continuous synchronization also require Delta Change Data
# MAGIC Feed to be enabled on the source table.

# COMMAND ----------

source_columns = {field.name for field in spark.table(source_table).schema.fields}
if "id" not in source_columns:
    raise ValueError(f"{source_table} must contain an 'id' primary-key column.")
if "embedding" not in source_columns:
    raise ValueError(f"{source_table} has no 'embedding' column. Run Lecture 2.4 first.")

spark.sql(
    f"ALTER TABLE {source_table} "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

logger.info(f"Validated source table and enabled Change Data Feed: {source_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create the Synced Table
# MAGIC
# MAGIC The first run provisions the target and performs an initial
# MAGIC synchronization. Later runs inspect the existing synced table instead.
# MAGIC
# MAGIC - `type_overrides` syncs `embedding` as a native pgvector `vector(1024)`
# MAGIC   column. Without it, Delta `ARRAY` columns land in Postgres as `JSONB`.
# MAGIC - `new_pipeline_spec` sets the usage policy on the sync pipeline and the
# MAGIC   catalog/schema where the pipeline keeps its checkpoints and event logs.
# MAGIC - `TRIGGERED` is useful for a classroom because changes are synchronized
# MAGIC   when requested. Use `SNAPSHOT` for a complete replacement on each run, or
# MAGIC   `CONTINUOUS` to replicate the source continuously.
# MAGIC
# MAGIC Creating a synced table does not install Postgres extensions, so we enable
# MAGIC pgvector first: the `vector` type override needs it.

# COMMAND ----------

# The manager obtains a short-lived Lakebase database credential through the
# Databricks SDK. It does not store a password in the notebook.
pgvector = LakebasePgVectorManager(config=cfg, workspace=w)
pgvector.ensure_extension()

# COMMAND ----------

from databricks.sdk.service.postgres import (
    NewPipelineSpec,
    SyncedTable,
    SyncedTableSyncedTableSpec,
    SyncedTableSyncedTableSpecPgSpecificType,
    SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy,
    SyncedTableSyncedTableSpecTypeOverride,
)

try:
    synced = w.postgres.get_synced_table(name=f"synced_tables/{synced_table}")
    logger.info(f"Synced Table already exists: {synced_table}")
except Exception:
    if not RUN_CREATE_SYNC:
        logger.info(
            "Synced Table is not created. Set RUN_CREATE_SYNC=True and rerun "
            "this cell to provision it."
        )
    else:
        synced = w.postgres.create_synced_table(
            synced_table=SyncedTable(
                spec=SyncedTableSyncedTableSpec(
                    source_table_full_name=source_table,
                    branch=postgres_branch,
                    postgres_database="databricks_postgres",
                    primary_key_columns=["id"],
                    scheduling_policy=SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.TRIGGERED,
                    create_database_objects_if_missing=True,
                    new_pipeline_spec=NewPipelineSpec(
                        budget_policy_id=cfg.usage_policy_id,
                        storage_catalog=cfg.catalog,
                        storage_schema=cfg.schema,
                    ),
                    type_overrides=[
                        SyncedTableSyncedTableSpecTypeOverride(
                            column_name="embedding",
                            pg_type=SyncedTableSyncedTableSpecPgSpecificType.PG_SPECIFIC_TYPE_VECTOR,
                            size=1024,
                        )
                    ],
                )
            ),
            synced_table_id=synced_table,
        ).wait()
        logger.info(f"Created Synced Table: {synced_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Check synchronization status
# MAGIC
# MAGIC Wait until the Synced Table is online before querying it. The destination
# MAGIC is read-only: write changes to `mra_chunks_table`, not to the Lakebase
# MAGIC copy. A `TRIGGERED` table only syncs later source changes when a sync is
# MAGIC started: **Sync now** on the synced table's Overview tab, or a Lakeflow Jobs
# MAGIC *Database Table Sync* task with a table update trigger or a schedule.

# COMMAND ----------

try:
    status = w.postgres.get_synced_table(name=f"synced_tables/{synced_table}").status
    logger.info(f"State: {status.detailed_state} | {status.message}")
except Exception as error:
    logger.info(f"Synced Table status is not available yet: {error}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sync now from code
# MAGIC
# MAGIC A Synced Table runs on a Lakeflow pipeline, and its status carries that
# MAGIC pipeline's id. Starting a pipeline update runs a sync, like **Sync now** in
# MAGIC the UI. This uses the general Pipelines API: the synced tables documentation
# MAGIC only describes the UI button and the Lakeflow Jobs task.

# COMMAND ----------

if RUN_SYNC_NOW:
    pipeline_id = w.postgres.get_synced_table(
        name=f"synced_tables/{synced_table}"
    ).status.pipeline_id
    update = w.pipelines.start_update(pipeline_id=pipeline_id)
    logger.info(f"Started sync: pipeline {pipeline_id}, update {update.update_id}")
else:
    logger.info("Sync skipped. Set RUN_SYNC_NOW=True to sync source changes now.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Query the Lakebase mirror
# MAGIC
# MAGIC This table is in Lakebase Postgres, not in the Spark SQL catalog. Query it
# MAGIC through the Lakebase connection after it reaches `ONLINE`. Thanks to the
# MAGIC type override, `embedding` is a pgvector `vector`.

# COMMAND ----------

with pgvector.connection() as conn:
    rows = conn.execute(
        f"""SELECT id, paper_id, title, corpus, year, pg_typeof(embedding) AS embedding_type, text
            FROM {pgvector.table_name}
            LIMIT 5"""
    ).fetchall()

for row in rows:
    logger.info(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Index the Synced Table for retrieval
# MAGIC
# MAGIC A Synced Table accepts `SELECT`, `CREATE INDEX`, and `DROP TABLE`, but no
# MAGIC writes. The table is owned by an internal `databricks_writer_<dbid>` role; if
# MAGIC `CREATE INDEX` is denied, a `databricks_superuser` must grant you index
# MAGIC permissions. We add:
# MAGIC
# MAGIC - an HNSW index using cosine distance (`<=>`) on `embedding`
# MAGIC - a GIN index on `to_tsvector('english', title || ' ' || text)` for keyword
# MAGIC   search; queries use the same expression, so Postgres can use the index
# MAGIC
# MAGIC The call is idempotent: rerun it if the indexes are missing, for example
# MAGIC after the Synced Table was recreated.

# COMMAND ----------

pgvector.ensure_indexes()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Semantic search
# MAGIC
# MAGIC The query is embedded with the same endpoint as the documents. PostgreSQL
# MAGIC ranks rows by cosine distance and returns the highest-scoring chunks.

# COMMAND ----------

query = "What are the problems in credit risk modeling?"
results = pgvector.search(query=query, num_results=5)

logger.info(f"Query: {query}")
for position, row in enumerate(results, 1):
    logger.info(
        f"{position}. {row.get('title', 'N/A')} | "
        f"paper_id: {row.get('paper_id', 'N/A')} | "
        f"score={float(row['score']):.4f}"
    )
    logger.info(f"   {row.get('text', '')[:250]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Semantic search with metadata filtering

# COMMAND ----------

query = "data requirements according to Us regulation"
year = 2026
corpus = "regulatory"
filtered_results = pgvector.search(query=query, num_results=3, corpus=corpus, year=year)

logger.info(f"Query: {query} | corpus={corpus} | year={year}")
for position, row in enumerate(filtered_results, 1):
    logger.info(
        f"{position}. {row.get('title', 'N/A')} | "
        f"year={row.get('year', 'N/A')} | score={float(row['score']):.4f}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Hybrid search: vector similarity + full-text ranking
# MAGIC
# MAGIC Hybrid search combines semantic similarity with PostgreSQL's
# MAGIC `ts_rank_cd()` full-text score. This helps with exact terms, acronyms, and
# MAGIC model names that semantic search alone can underweight.

# COMMAND ----------

query = "can I use LightGBM or Neural Networks in credit models in US"
hybrid_results = pgvector.hybrid_search(query=query, num_results=5)

logger.info(f"Hybrid query: {query}")
for position, row in enumerate(hybrid_results, 1):
    logger.info(
        f"{position}. {row.get('title', 'N/A')} | "
        f"score={float(row['score']):.4f} | "
        f"vector={float(row['vector_score']):.4f} | "
        f"text={float(row['text_score']):.4f}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Inspect the indexes

# COMMAND ----------

with pgvector.connection() as conn:
    table_count = conn.execute(f"SELECT count(*) FROM {pgvector.table_name}").fetchone()[0]
    indexes = conn.execute(
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = %s AND tablename = %s
        ORDER BY indexname
        """,
        (cfg.schema, pgvector.synced_table_name),
    ).fetchall()

logger.info(f"Synced rows: {table_count}")
for index_name, definition in indexes:
    logger.info(f"{index_name}: {definition}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC 1. Keep `mra_chunks_table`, with its `embedding` column, as the source of truth.
# MAGIC 2. A Synced Table replicates it into Lakebase; never write to the copy.
# MAGIC 3. Index the copy: HNSW (cosine) on `embedding`, GIN on the full-text expression.
# MAGIC 4. Combine vector, metadata, and full-text ranking in SQL.
# MAGIC
# MAGIC For a new embedding model, re-embed the Delta table, refresh the Synced Table,
# MAGIC and make sure the dimension matches the `vector(1024)` type override.