from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MANGA_LEARNING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8765
    cache_db_path: str = Field(default="cache.db")
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    ocr_backend: str = "mokuro"
    ocr_force_cpu: bool = True
    ocr_max_image_bytes: int = 50 * 1024 * 1024
    ocr_concurrency: int = 1
    ocr_cache_ttl_seconds: int = 24 * 60 * 60
    manga_server_url: str = ""
    ocr_page_ratio_min: float = 1.25
    ocr_page_ratio_max: float = 1.85
    ocr_min_pages: int = 8
    ocr_max_pages: int = 25
    ocr_virtual_page_overlap: float = 0.1
    ocr_iou_dedup_threshold: float = 0.5

    anki_connect_url: str = "http://127.0.0.1:8765"
    anki_enabled: bool = False

    ai_provider: str = "minimax"
    ai_enabled: bool = False

    ai_minimax_api_key: str = ""
    ai_minimax_base_url: str = "https://api.minimax.io/v1"
    ai_minimax_model: str = "MiniMax-M3"

    ai_openai_api_key: str = ""
    ai_openai_model: str = "gpt-4o-mini"
    ai_openai_base_url: str = "https://api.openai.com/v1"

    ai_ollama_url: str = "http://127.0.0.1:11434"
    ai_ollama_model: str = "llama3.1"


def get_settings() -> Settings:
    return Settings()