"""
Central configuration for the RAG Knowledge Assistant.

All tunables (RAG defaults, storage paths, API keys) live here and are
sourced from environment variables / a .env file via pydantic-settings.
No component elsewhere should read os.environ directly -- everything
routes through this single `settings` object so behaviour is easy to
reason about and change in one place.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (backend/app/config.py -> repo root).
# Anchoring storage paths here -- rather than leaving them as bare relative
# paths -- means `data/uploads` always resolves to the same physical
# location regardless of the current working directory the app is
# launched from (e.g. `cd backend && uvicorn ...` vs running from repo root).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed application settings.

    Every field has a sane default so the app can start even without a
    .env file (except GROQ_API_KEY, which is required only at the point
    generation is actually attempted -- not at import/startup time, so
    the rest of the pipeline (upload, chunk, embed, search) can be
    developed and tested without a Groq key).
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (Groq) -----------------------------------------------------
    groq_api_key: str = ""
    # Kept as a plain string (not an Enum) because Groq's available model
    # ids change over time -- llama-3.3-70b-versatile, for example, was
    # deprecated by Groq in 2026 in favor of openai/gpt-oss-120b, which
    # is the current default here. Check
    # https://console.groq.com/docs/models for the live list before
    # assuming any hardcoded id still works.
    groq_model: str = "openai/gpt-oss-120b"

    # --- Embeddings -------------------------------------------------------
    # Open-source, local, no API key needed. See README "Embedding Model"
    # section for why this specific model was chosen over alternatives.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- RAG pipeline defaults ---------------------------------------------
    # These are *defaults* the UI pre-fills; every request can still
    # override chunk_size / chunk_overlap / top_k explicitly.
    default_chunk_size: int = 800
    default_chunk_overlap: int = 150
    default_top_k: int = 5

    # --- Storage ------------------------------------------------------------
    # Anchored to the project root by default (see _PROJECT_ROOT above) so
    # the app behaves the same no matter where it's launched from; override
    # with an absolute path in production (e.g. a mounted volume).
    upload_dir: Path = _PROJECT_ROOT / "data" / "uploads"
    index_dir: Path = _PROJECT_ROOT / "data" / "indexes"
    max_file_size_mb: int = 20

    # --- CORS -----------------------------------------------------------------
    # Comma-separated list of allowed origins, or "*" for development.
    cors_origins: str = "*"

    def cors_origin_list(self) -> list[str]:
        """Return cors_origins as a list, splitting on commas."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def ensure_storage_dirs(self) -> None:
        """Create upload/index directories if they don't exist yet."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


# Single shared settings instance, imported everywhere else as:
#   from app.config import settings
settings = Settings()
settings.ensure_storage_dirs()
