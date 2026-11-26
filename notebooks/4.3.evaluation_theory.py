# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 4.3: GenAI Evaluation Theory
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Why evaluation matters for GenAI
# MAGIC - Types of evaluation metrics
# MAGIC - MLflow evaluation framework
# MAGIC - Guidelines vs Judges
# MAGIC - Custom scorers
# MAGIC - Judge alignment with human feedback

# COMMAND ----------

import mlflow
from mlflow.genai.judges import make_judge
from typing import Literal
from pyspark.sql import SparkSession

from arxiv_curator.config import load_config, get_env
from arxiv_curator.evaluation import (
    polite_tone_guideline,
    hook_in_post_guideline,
    scope_guideline,
    word_count_check,
    mentions_papers
)
import os
from dotenv import load_dotenv

from loguru import logger

# COMMAND ----------

# Setup
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Why Evaluation Matters for GenAI
# MAGIC
# MAGIC Traditional ML evaluation (accuracy, F1, etc.) doesn't work well for GenAI because:
# MAGIC
# MAGIC ### Challenges:
# MAGIC - **Open-ended outputs**: No single "correct" answer
# MAGIC - **Subjective quality**: What's "good" varies by use case
# MAGIC - **Multiple dimensions**: Accuracy, tone, style, safety, etc.
# MAGIC - **Context-dependent**: Same output may be good/bad in different contexts
# MAGIC
# MAGIC ### Why Evaluate?
# MAGIC 1. **Quality assurance**: Ensure outputs meet standards
# MAGIC 2. **Regression detection**: Catch degradation over time
# MAGIC 3. **Model comparison**: Choose the best model/prompt
# MAGIC 4. **Continuous improvement**: Identify areas to improve
# MAGIC 5. **Trust & safety**: Detect harmful outputs

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Types of Evaluation Metrics
# MAGIC
# MAGIC ### A. Automated Metrics
# MAGIC
# MAGIC | Metric | What it Measures | Use Case |
# MAGIC |--------|-----------------|----------|
# MAGIC | **BLEU** | N-gram overlap | Translation, summarization |
# MAGIC | **ROUGE** | Recall of n-grams | Summarization |
# MAGIC | **Perplexity** | Model confidence | Language modeling |
# MAGIC | **Exact Match** | Perfect match | QA, classification |
# MAGIC | **F1 Score** | Token overlap | QA, NER |
# MAGIC
# MAGIC ### B. LLM-as-Judge Metrics
# MAGIC
# MAGIC | Metric | What it Measures | Use Case |
# MAGIC |--------|-----------------|----------|
# MAGIC | **Relevance** | Answer relevance to question | QA, search |
# MAGIC | **Faithfulness** | Grounded in context | RAG systems |
# MAGIC | **Coherence** | Logical flow | Long-form generation |
# MAGIC | **Tone** | Professional, friendly, etc. | Customer service |
# MAGIC | **Safety** | Harmful content detection | All applications |
# MAGIC
# MAGIC ### C. Human Evaluation
# MAGIC
# MAGIC - Most reliable but expensive
# MAGIC - Used to validate automated metrics
# MAGIC - Essential for nuanced quality assessment

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. MLflow Evaluation Framework
# MAGIC
# MAGIC MLflow provides a comprehensive framework for GenAI evaluation:
# MAGIC
# MAGIC ```python
# MAGIC results = mlflow.genai.evaluate(
# MAGIC     data=eval_data,              # Test cases
# MAGIC     predict_fn=my_model,         # Model to evaluate
# MAGIC     scorers=[scorer1, scorer2],  # Evaluation metrics
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### Key Components:
# MAGIC 1. **Data**: Test cases with inputs and optionally expected outputs
# MAGIC 2. **Predict Function**: Your model/agent
# MAGIC 3. **Scorers**: Metrics to evaluate outputs
# MAGIC 4. **Results**: Detailed evaluation results

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Guidelines - Binary Pass/Fail

# COMMAND ----------

# Set experiment
mlflow.set_experiment(cfg.experiment_name)

# Using pre-defined Guidelines scorer from arxiv_curator.evaluation
# The polite_tone_guideline is already imported from our package
# It includes guidelines for polite and professional tone

logger.info("Using Guidelines Scorer from arxiv_curator.evaluation:")
logger.info(f"  Name: {polite_tone_guideline.name}")
logger.info(f"  Type: Binary (Pass/Fail)")
logger.info(f"  Guidelines: {len(polite_tone_guideline.guidelines)} rules")
logger.info(f"Also available from package:")
logger.info(f"  - hook_in_post_guideline: Checks for engaging hooks")
logger.info(f"  - scope_guideline: Ensures responses stay on topic")
logger.info(f"  - word_count_check: Custom scorer for word count")
logger.info(f"  - mentions_papers: Checks if response mentions research papers")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Guidelines

# COMMAND ----------

# Test data
test_data = [
    {
        "inputs": {"question": "How do I deploy a model?"},
        "outputs": "Just figure it out yourself, it's not that hard.",
    },
    {
        "inputs": {"question": "How do I deploy a model?"},
        "outputs": "I'd be happy to help you deploy your model! Here are the steps...",
    },
]

# Evaluate
results = mlflow.genai.evaluate(
    data=test_data,
    scorers=[polite_tone_guideline]
)

logger.info("Evaluation Results:")
logger.info("=" * 80)
display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Judges - Scored Evaluation

# COMMAND ----------

# Create a Judge with numeric scoring
quality_judge = make_judge(
    name="response_quality",
    instructions=(
        "Evaluate the quality of the response in {{ outputs }} to the question in {{ inputs }}. "
        "Score from 1 to 5:\n"
        "1 - Completely unhelpful or incorrect\n"
        "2 - Partially helpful but missing key information\n"
        "3 - Adequate response with some useful information\n"
        "4 - Good response with clear and helpful information\n"
        "5 - Excellent response that is comprehensive and well-explained"
    ),
    model=f"databricks:/{cfg.llm_endpoint}",
    feedback_value_type=int,
)

logger.info("Judge Scorer Created:")
logger.info(f"  Name: {quality_judge.name}")
logger.info(f"  Type: Scored (1-5)")
logger.info(f"  Judge Model: {cfg.llm_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Judge

# COMMAND ----------

# Test data with varying quality
judge_test_data = [
    {
        "inputs": {"question": "What is machine learning?"},
        "outputs": "It's computers learning stuff.",
    },
    {
        "inputs": {"question": "What is machine learning?"},
        "outputs": "Machine learning is a subset of AI where algorithms learn patterns from data to make predictions or decisions without being explicitly programmed.",
    },
]

# Evaluate
judge_results = mlflow.genai.evaluate(
    data=judge_test_data,
    scorers=[quality_judge]
)

logger.info("Judge Evaluation Results:")
logger.info("=" * 80)
display(judge_results.tables['eval_results'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Custom Code-Based Scorers

# COMMAND ----------

@mlflow.genai.scorer
def word_count_check(outputs: list) -> bool:
    """Check that the output is under 350 words."""
    text = outputs[0].get("text", "") if isinstance(outputs[0], dict) else str(outputs[0])
    word_count = len(text.split())
    return word_count < 350

@mlflow.genai.scorer
def has_code_example(outputs: list) -> bool:
    """Check if output contains a code example."""
    text = outputs[0].get("text", "") if isinstance(outputs[0], dict) else str(outputs[0])
    return "```" in text or "python" in text.lower()

@mlflow.genai.scorer
def response_length_score(outputs: list) -> float:
    """Score based on response length (0-1)."""
    text = outputs[0].get("text", "") if isinstance(outputs[0], dict) else str(outputs[0])
    word_count = len(text.split())
    
    # Ideal range: 50-200 words
    if 50 <= word_count <= 200:
        return 1.0
    elif word_count < 50:
        return word_count / 50  # Penalize too short
    else:
        return max(0.0, 1.0 - (word_count - 200) / 200)  # Penalize too long

logger.info("Custom Scorers Created:")
logger.info("  1. word_count_check (boolean)")
logger.info("  2. has_code_example (boolean)")
logger.info("  3. response_length_score (float 0-1)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Custom Scorers

# COMMAND ----------

custom_test_data = [
    {
        "inputs": {"question": "How to use Python?"},
        "outputs": "Here's how:\n```python\nprint('Hello')\n```\nThis prints Hello to the console.",
    },
    {
        "inputs": {"question": "How to use Python?"},
        "outputs": "Python is a programming language. " * 100,  # Very long
    },
]

# Evaluate with custom scorers
custom_results = mlflow.genai.evaluate(
    data=custom_test_data,
    scorers=[word_count_check, has_code_example, response_length_score]
)

logger.info("Custom Scorer Results:")
logger.info("=" * 80)
display(custom_results.tables['eval_results'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Categorical Judges

# COMMAND ----------

# Judge with categorical output
sentiment_judge = make_judge(
    name="response_sentiment",
    instructions=(
        "Analyze the sentiment of the response in {{ outputs }}. "
        "Classify as: 'positive', 'neutral', or 'negative'"
    ),
    feedback_value_type=Literal["positive", "neutral", "negative"],
    model=f"databricks:/{cfg.llm_endpoint}",
)

logger.info("Categorical Judge Created:")
logger.info(f"  Name: {sentiment_judge.name}")
logger.info(f"  Categories: positive, neutral, negative")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Combining Multiple Scorers

# COMMAND ----------

# Combine different types of scorers
all_scorers = [
    polite_tone_guideline,      # Binary guideline
    quality_judge,              # Numeric judge (1-5)
    word_count_check,           # Boolean custom
    response_length_score,      # Float custom (0-1)
    sentiment_judge,            # Categorical judge
]

comprehensive_test_data = [
    {
        "inputs": {"question": "Explain transformers"},
        "outputs": "Transformers are a neural network architecture that uses self-attention mechanisms to process sequential data. They've revolutionized NLP by enabling models like BERT and GPT.",
    },
]

# Evaluate with all scorers
comprehensive_results = mlflow.genai.evaluate(
    data=comprehensive_test_data,
    scorers=all_scorers
)

logger.info("Comprehensive Evaluation Results:")
logger.info("=" * 80)
comprehensive_results

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Judge Alignment with Human Feedback

# COMMAND ----------

# MAGIC %md
# MAGIC ### The Problem:
# MAGIC - LLM judges may not align with human preferences
# MAGIC - Different judges may score differently
# MAGIC - Need to calibrate judges to match human judgment
# MAGIC
# MAGIC ### The Solution: SIMBA Alignment
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.judges.optimizers import SIMBAAlignmentOptimizer
# MAGIC
# MAGIC # Get traces with both judge and human feedback
# MAGIC traces_with_feedback = get_traces_with_human_feedback()
# MAGIC
# MAGIC # Create optimizer
# MAGIC optimizer = SIMBAAlignmentOptimizer(model="databricks:/my-llm")
# MAGIC
# MAGIC # Align judge to human feedback
# MAGIC aligned_judge = my_judge.align(optimizer, traces_with_feedback)
# MAGIC ```
# MAGIC
# MAGIC ### How it Works:
# MAGIC 1. Collect human feedback on outputs
# MAGIC 2. Compare judge scores with human scores
# MAGIC 3. Optimize judge instructions to match human preferences
# MAGIC 4. Use aligned judge for future evaluations

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: Judge Alignment (Conceptual)

# COMMAND ----------

# Note: This is a conceptual example
# In practice, you would need actual traces with human feedback

logger.info("Judge Alignment Process:")
logger.info("=" * 80)
logger.info("1. Create initial judge")
logger.info("2. Evaluate test cases")
logger.info("3. Collect human feedback on same cases")
logger.info("4. Use SIMBAAlignmentOptimizer to align judge")
logger.info("5. Use aligned judge for production evaluation")
logger.info("Benefits:")
logger.info("  - Judge scores match human preferences")
logger.info("  - More reliable automated evaluation")
logger.info("  - Reduced need for human evaluation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Best Practices

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ Evaluation Best Practices:
# MAGIC
# MAGIC 1. **Use multiple scorers** for comprehensive evaluation
# MAGIC 2. **Combine automated and LLM judges** for balance
# MAGIC 3. **Create domain-specific guidelines** for your use case
# MAGIC 4. **Validate judges with human feedback** periodically
# MAGIC 5. **Track evaluation metrics over time** for regression detection
# MAGIC 6. **Use appropriate judge models** (not too small, not too expensive)
# MAGIC 7. **Test edge cases** in your evaluation data
# MAGIC 8. **Document evaluation criteria** clearly
# MAGIC 9. **Version your evaluation sets** for reproducibility
# MAGIC 10. **Align judges with human feedback** for production use
# MAGIC
# MAGIC ### ❌ Don't:
# MAGIC 1. Rely on a single metric
# MAGIC 2. Use only automated metrics for GenAI
# MAGIC 3. Ignore human feedback
# MAGIC 4. Evaluate on too few examples
# MAGIC 5. Forget to version evaluation data
# MAGIC 6. Use the same model as both generator and judge

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Evaluation Workflow

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recommended Workflow:
# MAGIC
# MAGIC ```
# MAGIC 1. Define Evaluation Criteria
# MAGIC    ├─ What makes a "good" output?
# MAGIC    ├─ What dimensions matter? (accuracy, tone, safety, etc.)
# MAGIC    └─ What are edge cases?
# MAGIC
# MAGIC 2. Create Evaluation Dataset
# MAGIC    ├─ Diverse test cases
# MAGIC    ├─ Edge cases
# MAGIC    └─ Representative of production
# MAGIC
# MAGIC 3. Choose Scorers
# MAGIC    ├─ Guidelines for binary checks
# MAGIC    ├─ Judges for nuanced scoring
# MAGIC    └─ Custom scorers for specific needs
# MAGIC
# MAGIC 4. Run Evaluation
# MAGIC    ├─ Evaluate baseline
# MAGIC    ├─ Evaluate improvements
# MAGIC    └─ Compare models/prompts
# MAGIC
# MAGIC 5. Analyze Results
# MAGIC    ├─ Identify failure patterns
# MAGIC    ├─ Find improvement opportunities
# MAGIC    └─ Validate with human review
# MAGIC
# MAGIC 6. Iterate
# MAGIC    ├─ Improve model/prompt
# MAGIC    ├─ Re-evaluate
# MAGIC    └─ Deploy if better
# MAGIC ```

# COMMAND ----------
