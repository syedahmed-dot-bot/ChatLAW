# config/settings.py
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    APP_NAME: str = "ChatLAW"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    CORS_ALLOW_ORIGINS: list[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]

    DATA_DIR: Path = Field(default_factory=lambda: Path("data").resolve())
    CORPUS_DIR: Path = Field(default_factory=lambda: Path("data/corpus").resolve())
    INDEX_DIR: Path = Field(default_factory=lambda: Path("data/index").resolve())

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 6

    LLM_PROVIDER: str = "none"
    LLM_MODEL: str = "gpt-4o-mini"
    MAX_TOKENS: int = 800

    class Config:
        env_file = ".env"

settings = Settings()
