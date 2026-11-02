# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC ## 1. Why Chunking Matters
# MAGIC
# MAGIC **Chunking** is the process of breaking documents into smaller pieces for:
# MAGIC
# MAGIC 1. **Embedding Generation**: Most embedding models have token limits (512-8192 tokens)
# MAGIC 2. **Retrieval Precision**: Smaller chunks = more precise retrieval
# MAGIC 3. **Context Window**: LLMs have limited context windows
# MAGIC 4. **Cost Optimization**: Fewer tokens = lower costs
# MAGIC
# MAGIC ### The Chunking Trade-off
# MAGIC
# MAGIC - **Large chunks**: More context, but less precise retrieval
# MAGIC - **Small chunks**: More precise, but may lose context
# MAGIC
# MAGIC **Optimal chunk size**: 256-512 tokens for most use cases

# COMMAND ----------

# MAGIC %pip install --force-reinstall /Volumes/llmops_dev/lennon_jcruz/packages/python/model_risk_agent-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import col, concat_ws, explode, udf
import re
import json

from model_risk_agent.config import load_config, get_env

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env()
cfg = load_config("../project_config.yml", env)
catalog = cfg.catalog
schema = cfg.schema

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Chunking Strategies Overview
# MAGIC
# MAGIC ### Strategy 1: Fixed-Size Chunking
# MAGIC - Split by character count or token count
# MAGIC - Simple and fast
# MAGIC - May break sentences/paragraphs
# MAGIC
# MAGIC ### Strategy 2: Sentence-Based Chunking
# MAGIC - Split on sentence boundaries
# MAGIC - Preserves semantic units
# MAGIC - Variable chunk sizes
# MAGIC
# MAGIC ### Strategy 3: Paragraph-Based Chunking
# MAGIC - Split on paragraph boundaries
# MAGIC - Larger semantic units
# MAGIC - Better for documents with clear structure
# MAGIC
# MAGIC ### Strategy 4: Semantic Chunking
# MAGIC - Use AI to identify topic boundaries
# MAGIC - Most intelligent but slowest
# MAGIC - Best for complex documents
# MAGIC
# MAGIC ### Strategy 5: AI Parse Documents (Databricks)
# MAGIC - AI identifies document structure
# MAGIC - Extracts elements (text, tables, etc.)
# MAGIC - Each element becomes a chunk
# MAGIC - **This is what we'll use!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Chunks from Lecture 2.2
# MAGIC
# MAGIC We'll use the chunks already extracted and cleaned in Lecture 2.2.

# COMMAND ----------

MRA_CHUNK_TABLE = "llmops_dev.lennon_jcruz.mra_chunks_table"
# Load chunks from the arxiv_chunks_table created in Lecture 2.2
chunks_df = spark.table(MRA_CHUNK_TABLE)

logger.info(f"Total chunks available: {chunks_df.count()}")
chunks_df.show(20, truncate=50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Analyze Chunk Statistics
# MAGIC
# MAGIC Let's analyze the chunks created by AI Parse Documents:

# COMMAND ----------

# Calculate chunk statistics
chunk_stats = chunks_df.select(
    F.avg(F.length(col("text"))).alias("avg_length"),
    F.min(F.length(col("text"))).alias("min_length"),
    F.max(F.length(col("text"))).alias("max_length"),
    F.count("*").alias("total_chunks")
).collect()[0]

logger.info(f"Chunk Statistics:")
logger.info(f"  Total chunks: {chunk_stats['total_chunks']}")
logger.info(f"  Average length: {chunk_stats['avg_length']:.0f} characters")
logger.info(f"  Min length: {chunk_stats['min_length']} characters")
logger.info(f"  Max length: {chunk_stats['max_length']} characters")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Alternative Chunking Strategies
# MAGIC
# MAGIC While AI Parse Documents provides intelligent chunking, let's explore other strategies:

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 1: Fixed-Size Chunking
# MAGIC
# MAGIC Split text into fixed-size chunks with optional overlap:

# COMMAND ----------

def fixed_size_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Create fixed-size chunks with overlap.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks

# COMMAND ----------

# Example: Apply fixed-size chunking to a sample chunk
sample_text = chunks_df.withColumn("max_length", F.length(col("text")))
display(sample_text.orderBy(col("max_length").desc()).limit(10))

sample_text = chunks_df.filter((col("paper_id") == "2608.24582v1") & (col("chunk_id") == 16)).select("text").collect()[0][0]
fixed_chunks = fixed_size_chunking(sample_text, chunk_size=500, overlap=50)

logger.info(f"Original text length: {len(sample_text)} characters")
logger.info(f"Number of fixed-size chunks: {len(fixed_chunks)}")
logger.info(f"\nFirst chunk preview:")
logger.info(fixed_chunks[0][:200] + "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 2: Sentence-Based Chunking
# MAGIC
# MAGIC Split on sentence boundaries using regex:

# COMMAND ----------

def sentence_chunking(text: str, max_sentences: int = 5) -> list[str]:
    """Create chunks based on sentence boundaries.

    Args:
        text: Text to chunk
        max_sentences: Maximum sentences per chunk

    Returns:
        List of text chunks
    """
    # Simple sentence splitter (can be improved with spaCy/NLTK)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= max_sentences:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Add remaining sentences
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# COMMAND ----------

# Example: Apply sentence-based chunking
sentence_chunks = sentence_chunking(sample_text, max_sentences=5)

logger.info(f"Number of sentence-based chunks: {len(sentence_chunks)}")
logger.info(f"\nFirst chunk preview:")
logger.info(sentence_chunks[0][:200] + "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Chunk Size Recommendations
# MAGIC
# MAGIC Based on your use case:
# MAGIC
# MAGIC | Use Case | Recommended Size | Reasoning |
# MAGIC |----------|-----------------|-----------|
# MAGIC | **Question Answering** | 256-512 tokens | Precise retrieval, focused answers |
# MAGIC | **Summarization** | 512-1024 tokens | More context needed |
# MAGIC | **Semantic Search** | 256-512 tokens | Balance between precision and context |
# MAGIC | **Code Search** | 100-200 tokens | Function/class level granularity |
# MAGIC
# MAGIC **Token estimation**: ~4 characters = 1 token (English text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Best Practices

# COMMAND ----------

# MAGIC %md
# MAGIC ### Do:
# MAGIC 1. **Clean text** before chunking (remove extra whitespace, fix hyphenation)
# MAGIC 2. **Preserve metadata** (paper_id, title, authors, etc.)
# MAGIC 3. **Test different chunk sizes** for your specific use case
# MAGIC 4. **Use overlap** for better context (50-100 characters)
# MAGIC 5. **Monitor chunk quality** (length distribution, content quality)
# MAGIC
# MAGIC ### Don't:
# MAGIC 1. Split in the middle of sentences (unless using fixed-size)
# MAGIC 2. Ignore document structure (tables, lists, etc.)
# MAGIC 3. Forget to clean and normalize text
# MAGIC 4. Lose metadata during chunking
# MAGIC 5. Use the same chunk size for all document types
