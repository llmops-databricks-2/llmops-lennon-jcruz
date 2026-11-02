"""
arXiv API
   ↓ (download_and_store_papers)
PDFs in Volume + arxiv_papers table
   ↓ (parse_pdfs_with_ai)
ai_parsed_docs_table (JSON)
   ↓ (process_chunks)
arxiv_chunks_table (clean text + metadata + embedding)
   ├─ (AI Search Delta Sync index) (2.4 notebook)
   └─ (Lakebase Synced Table + LakebasePgVectorManager search) (2.5 notebook)
"""

import json
import os
import re
import time
from typing import Any
import requests
from dateutil.parser import parse
# from databricks.sdk import WorkspaceClient  # used only by the Gateway _embed_texts path below
# from openai import OpenAI

import arxiv
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import (
    col,
    concat_ws,
    current_timestamp,
    explode,
    expr,
    udf,
)
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from model_risk_agent.config import ProjectConfig
from model_risk_agent.config import RegulatorySource, load_regulatory_sources

SOURCE_DOCUMENTS_SCHEMA = T.StructType(
    [
        T.StructField("paper_id", T.StringType(), False),
        T.StructField("title", T.StringType(), True),
        T.StructField("authors", T.ArrayType(T.StringType()), True),
        T.StructField("summary", T.StringType(), True),
        T.StructField("pdf_url", T.StringType(), True),
        T.StructField("published", T.LongType(), True),
        T.StructField("processed", T.LongType(), True),
        T.StructField("volume_path", T.StringType(), True),
        T.StructField("corpus", T.StringType(), True),
    ]
)

class DataProcessor:
    """
    DataProcessor handles the complete workflow of:
    - Downloading papers from arXiv
    - Storing paper metadata
    - Parsing PDFs with ai_parse_document
    - Extracting and cleaning text chunks
    - Saving chunks to Delta tables
    """

    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        """
        Initialize DataProcessor with Spark session and configuration.

        Args:
            spark: SparkSession instance
            config: ProjectConfig object with table configurations
        """
        self.spark = spark
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema
        self.volume = config.volume

        full_volume = f"{self.catalog}.{self.schema}.{self.volume}"  # UC name
        volume_root = f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

        self.spark.sql(
            f"CREATE VOLUME IF NOT EXISTS {full_volume}"
        )

        self.end = time.strftime("%Y%m%d%H%M", time.gmtime())
        # Fixed URL set; same docs every run
        self.regulatory_dir = f"{volume_root}/regulatory"
        # Incremental arXiv pulls; parse one batch at a time
        self.academic_dir = f"{volume_root}/academic/{self.end}"

        os.makedirs(self.regulatory_dir, exist_ok=True)
        os.makedirs(self.academic_dir, exist_ok=True)
        self.papers_table = f"{self.catalog}.{self.schema}.mra_source_documents"
        self.parsed_table = f"{self.catalog}.{self.schema}.mra_ai_parsed_docs"

    def _get_range_start(self) -> str:
        """
        Get start time range for arxiv paper search.
        If arxiv_papers table exists, uses max(processed) as start.
        Otherwise, uses 720 days ago as start.

        Returns:
            start string in "YYYYMMDDHHMM" format
        """
        days = 720  # first backfill

        if self.spark.catalog.tableExists(self.papers_table):
            result = self.spark.sql(f"""
                SELECT max(processed)
                FROM {self.papers_table}
                where corpus = 'academic'
            """).collect()
            max_processed = result[0][0]


            if max_processed is not None:
                start = str(max_processed)
                logger.info(f"Found existing arxiv_papers table. Starting from: {start}")
            else:
                start = time.strftime("%Y%m%d%H%M", time.gmtime(time.time() - 24 * 3600 * days))
                logger.info(
                    f"Table exists but no processed records. Starting from {days} days ago: {start}"
                )
        else:
            start = time.strftime("%Y%m%d%H%M", time.gmtime(time.time() - 24 * 3600 * days))
            logger.info(f"No existing arxiv_papers table. Starting from {days} days ago: {start}")
        return start

    def download_and_store_papers(
        self,
    ) -> list[dict] | None:
        """
        Download papers from arxiv and store metadata
        in arxiv_papers table.

        Returns:
            List of paper metadata dictionaries if papers were downloaded,
            otherwise None
        """
        start = self._get_range_start()

        existing_ids = set()
        if self.spark.catalog.tableExists(self.papers_table):
            rows = self.spark.sql(f"""
                SELECT paper_id
                FROM {self.papers_table}
                WHERE corpus = 'academic'
            """).collect()
            existing_ids = {r.paper_id for r in rows}

        # Search for papers in arxiv
        client = arxiv.Client()
        search = arxiv.Search(query=f'cat:q-fin.RM AND (all:"credit risk" OR all:"model risk") AND submittedDate:[{start} TO {self.end}]')
        papers = client.results(search)

        # Download papers and collect metadata
        records = []

        for paper in papers:
            paper_id = paper.get_short_id()

            if paper_id in existing_ids:
                logger.info(f"Skip {paper_id}: already in table")
                continue

            try:
                paper.download_pdf(dirpath=self.academic_dir, filename=f"{paper_id}.pdf")
                # Collect metadata
                records.append(
                    {
                        "paper_id": paper_id,
                        "title": paper.title,
                        "authors": [author.name for author in paper.authors],
                        "summary": paper.summary,
                        "pdf_url": paper.pdf_url,
                        "published": int(paper.published.strftime("%Y%m%d%H%M")),
                        "processed": int(self.end),
                        "volume_path": f"{self.academic_dir}/{paper_id}.pdf",
                        "corpus": "academic",
                    }
                )
            except Exception as e:
                logger.warning(f"Paper {paper_id} was not successfully processed. {e}")
            # Avoid hitting API rate limits
            time.sleep(3)

        # Only process if we have records
        if len(records) == 0:
            logger.info("No new papers found.")
            return None

        logger.info(f"Downloaded {len(records)} papers to {self.academic_dir}")

        metadata_df = self.spark.createDataFrame(records, schema=SOURCE_DOCUMENTS_SCHEMA).withColumn(
            "ingest_ts", current_timestamp()
        )

        # Create table if it doesn't exist
        metadata_df.write.format("delta").mode("append").saveAsTable(self.papers_table)
        return records

    def download_and_store_public_pdfs(
        self,
        regulatory_sources: list[RegulatorySource],
    ) -> list[dict] | None:
        """
        Download public PDFs given a list of URLs and store them in the volume. 
        This is used to download regulatory PDFs from the web.
        Args:
            urls: List of URLs to download
        Returns:
            List of file paths if PDFs were downloaded, otherwise None
        """


        existing_ids = set()
        if self.spark.catalog.tableExists(self.papers_table):
            rows = self.spark.sql(f"""
                SELECT paper_id
                FROM {self.papers_table}
                WHERE corpus = 'regulatory'
            """).collect()
            existing_ids = {r.paper_id for r in rows}


        records = []

        for source in regulatory_sources:
            url = source.url
            doc_id = source.doc_id
            issued = int(parse(source.issued).strftime("%Y%m%d%H%M")) if source.issued else None

            if doc_id in existing_ids:
                logger.info(f"Skip {doc_id}: already in table")
                continue

            try:
                response = requests.get(url)
                response.raise_for_status()
                file_path = os.path.join(self.regulatory_dir, f"{doc_id}.pdf")
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                    records.append(
                        {
                            "paper_id": doc_id,
                            "title": source.title,
                            "authors": [source.agency] if source.agency else [],
                            "summary": source.description,
                            "pdf_url": url,
                            "published": issued,
                            "processed": int(self.end),
                            "volume_path": file_path,
                            "corpus": "regulatory",
                        }
                    )
            except Exception as e:
                logger.warning(f"Regulatory source {source.doc_id} was not successfully processed: {e}")
            # Avoid hitting API rate limits
            time.sleep(3)

        # Only process if we have records
        if len(records) == 0:
            logger.info("No new regulatory sources found.")
            return None

        logger.info(f"Downloaded {len(records)} regulatory sources to {self.regulatory_dir}")

        metadata_df = self.spark.createDataFrame(records, schema=SOURCE_DOCUMENTS_SCHEMA).withColumn(
            "ingest_ts", current_timestamp()
        )

        # Create table if it doesn't exist
        metadata_df.write.format("delta").mode("append").saveAsTable(self.papers_table)
        return records

    def parse_pdfs_with_ai(self) -> None:
        """
        Parse PDFs using ai_parse_document and store in ai_parsed_docs table.

        """

        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.parsed_table} (
                path STRING,
                parsed_content STRING,
                processed LONG
            )
        """)

        paths = [
            row.volume_path
            for row in self.spark.sql(f"""
                SELECT p.volume_path
                FROM {self.papers_table} p
                LEFT ANTI JOIN {self.parsed_table} t
                ON p.volume_path = t.path
                WHERE p.volume_path IS NOT NULL
            """).collect()
        ]

        if not paths:
            logger.info("No unparsed PDFs found.")
            return

        (
            self.spark.read.format("binaryFile")
            .load(paths)
            .selectExpr(
                "path",
                "ai_parse_document(content) AS parsed_content",
                f"CAST({self.end} AS BIGINT) AS processed",
            )
            .write.mode("append")
            .saveAsTable(self.parsed_table)
        )

        logger.info(f"Parsed {len(paths)} PDFs into {self.parsed_table}")

    @staticmethod
    def _extract_chunks(parsed_content_json: str) -> list[tuple[str, str]]:
        """
        Extract chunks from parsed_content JSON.

        Args:
            parsed_content_json: JSON string containing
            parsed document structure

        Returns:
            List of tuples containing (chunk_id, content)
        """
        parsed_dict = json.loads(parsed_content_json)
        chunks = []

        for element in parsed_dict.get("document", {}).get("elements", []):
            if element.get("type") == "text":
                chunk_id = element.get("id", "")
                content = element.get("content", "")
                chunks.append((chunk_id, content))

        return chunks

    @staticmethod
    def _extract_paper_id(path: str) -> str:
        """
        Extract paper ID from file path.

        Args:
            path: File path (e.g., "/path/to/paper_id.pdf")

        Returns:
            Paper ID extracted from the path
        """
        return path.replace(".pdf", "").split("/")[-1]

    @staticmethod
    def _clean_chunk(text: str) -> str:
        """
        Clean and normalize chunk text
        Args:
            text: Raw text content

        Returns:
            Cleaned text content
        """
        # Fix hyphenation across line breaks:
        # "docu-\nments" => "documents"
        t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Collapse internal newlines into spaces
        t = re.sub(r"\s*\n\s*", " ", t)

        # Collapse repeated whitespace
        t = re.sub(r"\s+", " ", t)

        return t.strip()

    # def _embed_texts(self, texts: list[str]) -> list[list[float]]:
    #     """Gateway OpenAI client path — not used: ai_query does not support AI Gateway models."""
    #     workspace = WorkspaceClient()
    #     client = OpenAI(
    #         api_key=workspace.tokens.create(lifetime_seconds=3600).token_value,
    #         base_url=f"{workspace.config.host.rstrip('/')}/ai-gateway/mlflow/v1",
    #     )
    #
    #     model = self.cfg.embedding_endpoint
    #     out: list[list[float]] = []
    #     batch_size = 32
    #
    #     for i in range(0, len(texts), batch_size):
    #         batch = texts[i : i + batch_size]
    #         resp = client.embeddings.create(model=model, input=batch)
    #         out.extend([row.embedding for row in sorted(resp.data, key=lambda r: r.index)])
    #     return out

    def process_chunks(self) -> None:
        """
        Process parsed documents to extract, clean, and embed chunks.
        Reads from ai_parsed_docs table and saves to arxiv_chunks table.
        """
        logger.info(f"Processing parsed documents from {self.parsed_table} for end date {self.end}")

        df = self.spark.table(self.parsed_table).where(f"processed = {self.end}")

        # Define schema for the extracted chunks
        chunk_schema = ArrayType(
            StructType(
                [
                    StructField("chunk_id", StringType(), True),
                    StructField("content", StringType(), True),
                ]
            )
        )

        extract_chunks_udf = udf(self._extract_chunks, chunk_schema)
        extract_paper_id_udf = udf(self._extract_paper_id, StringType())
        clean_chunk_udf = udf(self._clean_chunk, StringType())

        metadata_df = self.spark.table(self.papers_table).select(
            col("paper_id"),
            col("title"),
            col("summary"),
            col("corpus"),
            concat_ws(", ", col("authors")).alias("authors"),
            (col("published") / 100000000).cast("int").alias("year"),
            ((col("published") % 100000000) / 1000000).cast("int").alias("month"),
            ((col("published") % 1000000) / 10000).cast("int").alias("day"),
        )

        # Create the transformed dataframe
        chunks_df = (
            df.withColumn("paper_id", extract_paper_id_udf(col("path")))
            .withColumn("chunks", extract_chunks_udf(col("parsed_content")))
            .withColumn("chunk", explode(col("chunks")))
            .select(
                col("paper_id"),
                col("chunk.chunk_id").alias("chunk_id"),
                clean_chunk_udf(col("chunk.content")).alias("text"),
                concat_ws("_", col("paper_id"), col("chunk.chunk_id")).alias("id"),
            )
            .join(metadata_df, "paper_id", "left")
        )

        # Generate embeddings with the configured model
        # (ai_query is used via expr: not every runtime exposes it in pyspark.sql.functions)
        endpoint = self.cfg.embedding_endpoint.replace("'", "''")
        chunks_df = chunks_df.withColumn(
            "embedding",
            expr(f"CAST(ai_query('{endpoint}', text) AS ARRAY<FLOAT>)"),
        )

        # Write to table (mergeSchema adds the embedding column to existing tables)
        mra_chunks_table = f"{self.catalog}.{self.schema}.mra_chunks_table"
        chunks_df.write.mode("append").option("mergeSchema", "true").saveAsTable(
            mra_chunks_table
        )
        logger.info(f"Saved chunks to {mra_chunks_table}")

        # Enable Change Data Feed
        self.spark.sql(f"""
            ALTER TABLE {mra_chunks_table}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        logger.info(f"Change Data Feed enabled for {mra_chunks_table}")

    def process_and_save(self, regulatory_sources_path: str | None = None) -> None:
        """
        Complete workflow: download papers, parse PDFs, and process chunks.

        Args:
            regulatory_sources_path: Path to regulatory_sources.yml. Defaults to
                ``cfg.regulatory_sources_path`` when set, else ``regulatory_sources.yml``
                (resolved from cwd / parents).
        """
        # Step 1: Download papers and store metadata
        self.download_and_store_papers()

        # Step 2: Download regulatory PDFs and store metadata
        sources_path = (
            regulatory_sources_path
            or self.cfg.regulatory_sources_path
            or "regulatory_sources.yml"
        )
        sources = load_regulatory_sources(sources_path)
        self.download_and_store_public_pdfs(sources)

        # Step 3: Parse PDFs with ai_parse_document
        self.parse_pdfs_with_ai()
        logger.info("Parsed documents.")

        # Step 4: Process chunks
        self.process_chunks()
        logger.info("Processing complete!")
