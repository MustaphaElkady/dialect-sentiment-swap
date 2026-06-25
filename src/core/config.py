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
    TRAINING_PREVIEW_OUTPUT_DIR: str
    TRAINING_PREVIEW_FILE: str
    TRAINING_PREVIEW_MAX_EXAMPLES: int
    CAUSAL_MODEL_NAME: str
    CAUSAL_OUTPUT_DIR: str
    CAUSAL_PREDICTIONS_FILE: str
    CAUSAL_MAX_EXAMPLES: int

    CAUSAL_MAX_NEW_TOKENS: int
    CAUSAL_TEMPERATURE: float
    CAUSAL_TOP_P: float

    
    LLAMAFACTORY_DATA_DIR: str = "data/llamafactory"
    LLAMAFACTORY_TRAIN_FILE: str = "sentiment_swap_train.json"
    LLAMAFACTORY_EVAL_FILE: str = "sentiment_swap_val.json"
    LLAMAFACTORY_DATASET_INFO_FILE: str = "dataset_info.json"

    FINETUNE_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
    FINETUNE_OUTPUT_DIR: str = "/content/drive/MyDrive/dialect-sentiment-swap/models/qwen2_5_7b_sentiment_swap_qlora"
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()