from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str
    APP_VERSION: str

    ENVIRONMENT: str
    PROJECT_ROOT: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()