# config/settings.py
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices

class Settings(BaseSettings):
    # pydantic-settings v2 config
    model_config = SettingsConfigDict(
        env_file=".env",        # load .env at project root
        extra="ignore",         # ignore unknown env vars (prevents extra_forbidden)
        case_sensitive=False,   # accept upper/lower env names
    )

    # App
    APP_NAME: str = "ChatLAW"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # CORS
    CORS_ALLOW_ORIGINS: list[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]

    # Paths (resolved to absolute to avoid cwd quirks)
    DATA_DIR: Path = Field(default_factory=lambda: Path("data").resolve())
    CORPUS_DIR: Path = Field(default_factory=lambda: Path("data/corpus").resolve())
    INDEX_DIR: Path = Field(default_factory=lambda: Path("data/index").resolve())

    # RAG
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 6

    # LLM
    LLM_PROVIDER: str = "groq"          # "groq" | "openai" | "mistral" | "none"
    LLM_MODEL: str = "llama3-8b-8192"
    MAX_TOKENS: int = 800

    # Provider keys — accept both upper/lower variants
    GROQ_API_KEY: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "groq_api_key"),
    )

settings = Settings()
