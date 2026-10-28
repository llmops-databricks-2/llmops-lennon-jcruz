"""Configuration management for Model Risk Agent."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """Project configuration model."""

    catalog: str = Field(..., description="Unity Catalog name")
    db_schema: str = Field(..., description="Schema name", alias="schema")
    volume: str = Field(..., description="Volume name")
    llm_endpoint: str = Field(..., description="AI Gateway chat model service name")
    embedding_endpoint: str = Field(..., description="AI Gateway embedding model service name")
    experiment_path: str = Field(..., description="MLflow experiment path for this environment")
    warehouse_id: str = Field(..., description="Warehouse ID")
    genie_space_id: str | None = Field(None, description="Genie space ID for MCP integration")
    system_prompt: str = Field(
        default="You are a helpful AI assistant that helps users find and understand research papers.",
        description="System prompt for the agent",
    )

    model_config = {"populate_by_name": True}

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file
            env: Environment name (dev, acc, prd)

        Returns:
            ProjectConfig instance

        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        if env not in config_data:
            raise ValueError(f"Environment '{env}' not found in config file")

        return cls(**config_data[env])

    @property
    def schema(self) -> str:
        """Alias for db_schema for backward compatibility."""
        return self.db_schema

    @property
    def full_schema_name(self) -> str:
        """Get fully qualified schema name."""
        return f"{self.catalog}.{self.db_schema}"

    @property
    def full_volume_path(self) -> str:
        """Get fully qualified volume path."""
        return f"{self.catalog}.{self.schema}.{self.volume}"

    
def _resolve_path(path: str, max_up: int = 3) -> Path:
    """Resolve a relative path from cwd or parent dirs (notebook-friendly)."""
    
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p

    current = Path.cwd()
    for _ in range(max_up):
        candidate = current / path
        if candidate.exists():
            return candidate
        current = current.parent
    return p


def load_config(config_path: str = "project_config.yml", env: str = "dev") -> ProjectConfig:
    """Load project configuration.

    Args:
        config_path: Path to configuration file
        env: Environment name

    Returns:
        ProjectConfig instance

    """
    return ProjectConfig.from_yaml(str(_resolve_path(config_path)), env)


def get_env() -> str:
    """Get current environment from environment variable.

    Returns:
        Environment name (dev, acc, or prd)

    """
    return os.getenv("ENV", "dev")

class RegulatorySource(BaseModel):
    """Regulatory source model."""

    doc_id: str
    title: str
    url: str
    description: str | None = None
    agency: str | None = None
    status: str | None = None
    topics: list[str] = []
    issued: str | None = None
    superseded_by: str | None = None

def load_regulatory_sources(path: str = "regulatory_sources.yml") -> list[RegulatorySource]:

    with open(_resolve_path(path)) as f:
        data = yaml.safe_load(f)
    return [RegulatorySource(**item) for item in data["sources"]]