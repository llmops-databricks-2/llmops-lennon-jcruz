from dataclasses import dataclass
from pathlib import Path
import re
from model_risk_agent.config import ProjectConfig

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

@dataclass(frozen=True)
class PersistedModelDoc:
    """Result of writing one assessment's model documentation."""
    run_id: str
    model_name: str
    markdown_path: str
    runs_dir: str

class ModelDocStore:
    """Persist per-run Markdown model documentation on a UC volume (audit artifact)."""

    def __init__(self, config: ProjectConfig, *, volume_root: str | None = None) -> None:
        self.config = config
        self.volume_root = volume_root or (
            f"/Volumes/{config.catalog}/{config.schema}/{config.model_doc_volume}"
        )

    @property
    def runs_dir(self) -> Path:
        """Directory that holds all per-run Markdown files."""
        return Path(self.volume_root) / "runs"

    def artifact_path(self, model_name: str, run_id: str) -> Path:
        """Build ``.../runs/{safe_model}_{run_id}.md``."""
        self._validate_run_id(run_id)
        safe_model = self._safe_model_name(model_name)
        return self.runs_dir / f"{safe_model}_{run_id}.md"

    def persist(
            self,
            run_id: str,
            markdown: str,
            *,
            model_name: str,
        ) -> PersistedModelDoc:
            """Write Markdown for one assessment and return the volume path."""
            text = (markdown or "").strip()
            if not text:
                raise ValueError("model documentation Markdown is required")

            if not (model_name or "").strip():
                raise ValueError("model_name is required")

            # Reject accidental path-as-content (common notebook mistake)
            if text.startswith("/Volumes/") or text.startswith("dbfs:"):
                raise ValueError(
                    "markdown looks like a path; use persist_from_path(...) instead"
                )

            self.runs_dir.mkdir(parents=True, exist_ok=True)
            md_path = self.artifact_path(model_name, run_id)
            md_path.write_text(text, encoding="utf-8")

            return PersistedModelDoc(
                run_id=run_id,
                model_name=self._safe_model_name(model_name),
                markdown_path=str(md_path),
                runs_dir=str(self.runs_dir),
            )

    def read_markdown(self, model_name: str, run_id: str) -> str:
        """Read a previously persisted model doc."""
        path = self.artifact_path(model_name, run_id)
        if not path.is_file():
            raise FileNotFoundError(f"No model documentation at {path}")
        return path.read_text(encoding="utf-8")

    def persist_from_path(
        self,
        run_id: str,
        source_path: str | Path,
        *,
        model_name: str,
    ) -> PersistedModelDoc:
        """Read a Markdown file, then persist an audit copy for this run."""
        path = Path(source_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model documentation not found: {path}")

        text = path.read_text(encoding="utf-8")
        return self.persist(run_id, text, model_name=model_name)

    def persist_from_bytes(
        self,
        run_id: str,
        data: bytes,
        *,
        model_name: str,
    ) -> PersistedModelDoc:
        """Persist Markdown from an uploaded file's bytes."""
        return self.persist(run_id, data.decode("utf-8"), model_name=model_name)

    @staticmethod
    def _safe_model_name(model_name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", model_name.strip())
        safe = safe.strip("_")
        if not safe:
            raise ValueError("model_name must contain letters, digits, _ or -")
        return safe

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not run_id or not _SAFE_RUN_ID.match(run_id):
            raise ValueError(
                "run_id must be 1–128 chars: letters, digits, . _ - (no path separators)"
            )