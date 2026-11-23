"""Lakebase session memory (chat history), separate from chunk search."""

from __future__ import annotations

import json
import os
import urllib.parse

import psycopg
from databricks.sdk import WorkspaceClient
from loguru import logger


class LakebaseMemory:
    def __init__(
        self,
        project_id: str,
        *,
        workspace: WorkspaceClient | None = None,
        table_name: str = "session_messages",
    ) -> None:
        self.project_id = project_id
        self.workspace = workspace or WorkspaceClient()
        self.table_name = table_name

    def _conn_string(self) -> str:
        w = self.workspace
        pg = w.postgres
        project_name = f"projects/{self.project_id}"
        branch = next(iter(pg.list_branches(parent=project_name)))
        endpoint = next(iter(pg.list_endpoints(parent=branch.name)))
        host = endpoint.status.hosts.host
        cred = pg.generate_database_credential(endpoint=endpoint.name)
        user = urllib.parse.quote_plus(
            os.environ.get("DATABRICKS_CLIENT_ID") or w.current_user.me().user_name
        )
        return (
            f"postgresql://{user}:{cred.token}@{host}:5432/"
            "databricks_postgres?sslmode=require"
        )

    def connection(self) -> psycopg.Connection:
        return psycopg.connect(self._conn_string())

    def ensure_schema(self) -> None:
        with self.connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_session_id
                ON {self.table_name}(session_id)
                """
            )
        logger.info("session memory table ready")

    def save_messages(self, session_id: str, messages: list[dict]) -> None:
        logger.info(
            "memory save session_id={} n_messages={} table={}",
            session_id,
            len(messages),
            self.table_name,
        )
        with self.connection() as conn:
            for msg in messages:
                conn.execute(
                    f"INSERT INTO {self.table_name} (session_id, message_data) "
                    "VALUES (%s, %s)",
                    (session_id, json.dumps(msg)),
                )

    def load_messages(self, session_id: str) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT message_data FROM {self.table_name}
                WHERE session_id = %s
                ORDER BY created_at ASC, id ASC
                """,
                (session_id,),
            ).fetchall()
        out = []
        for (data,) in rows:
            out.append(data if isinstance(data, dict) else json.loads(data))
        logger.info(
            "memory load session_id={} n_messages={} table={}",
            session_id,
            len(out),
            self.table_name,
        )
        return out