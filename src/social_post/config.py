from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    model: str = Field("groq/openai/gpt-oss-20b", alias="SOCIAL_POST_MODEL")
    groq_api_key: str = Field("", alias="GROQ_API_KEY")
    serper_api_key: str = Field("", alias="SERPER_API_KEY")
    rate_limit_max_retries: int = Field(
        12,
        ge=0,
        le=100,
        alias="SOCIAL_POST_RATE_LIMIT_MAX_RETRIES",
    )
    rate_limit_max_wait_seconds: float = Field(
        60.0,
        ge=1.0,
        le=300.0,
        alias="SOCIAL_POST_RATE_LIMIT_MAX_WAIT_SECONDS",
    )
    structured_output_retries: int = Field(
        2,
        ge=0,
        le=5,
        alias="SOCIAL_POST_STRUCTURED_OUTPUT_RETRIES",
    )

    data_dir: Path = Field(Path(".data"), alias="SOCIAL_POST_DATA_DIR")
    host: str = Field("127.0.0.1", alias="SOCIAL_POST_HOST")
    port: int = Field(8000, alias="SOCIAL_POST_PORT")

    footer: str = Field("", alias="SOCIAL_POST_FOOTER")
    primary_color: str = Field("#0A66C2", alias="SOCIAL_POST_PRIMARY_COLOR")
    accent_color: str = Field("#7C3AED", alias="SOCIAL_POST_ACCENT_COLOR")

    linkedin_client_id: str = Field("", alias="LINKEDIN_CLIENT_ID")
    linkedin_client_secret: str = Field("", alias="LINKEDIN_CLIENT_SECRET")
    linkedin_redirect_uri: str = Field(
        "http://127.0.0.1:8000/auth/linkedin/callback",
        alias="LINKEDIN_REDIRECT_URI",
    )
    linkedin_api_version: str = Field("202609", alias="LINKEDIN_API_VERSION")

    @property
    def workflows_dir(self) -> Path:
        return self.data_dir / "workflows"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def flow_db_path(self) -> Path:
        return self.data_dir / "flow_states.db"

    @property
    def linkedin_token_path(self) -> Path:
        return self.data_dir / "linkedin_token.json"

    def ensure_directories(self) -> None:
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    @property
    def generation_ready(self) -> bool:
        return bool(self.groq_api_key and self.serper_api_key)

    @property
    def linkedin_ready(self) -> bool:
        return bool(self.linkedin_client_id and self.linkedin_client_secret)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
