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

    
    ######### DATA ##############
    RAW_DATA_DIR: str
    PROCESSED_DATA_DIR: str

    RAW_TRAIN_FILE: str
    RAW_EVAL_FILE: str

    PROCESSED_TRAIN_FILE: str
    PROCESSED_EVAL_FILE: str

    DATASET_VALIDATION_REPORT_FILE: str
    TRAIN_MODEL_NAME: str

    MAX_SOURCE_LENGTH: int
    MAX_TARGET_LENGTH: int

    BASELINE_OUTPUT_DIR: str
    BASELINE_PREDICTIONS_FILE: str
    BASELINE_MAX_EXAMPLES: int

    GENERATION_MAX_NEW_TOKENS: int
    GENERATION_NUM_BEAMS: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()