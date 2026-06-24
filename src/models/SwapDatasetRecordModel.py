from src.models.BaseDataModel import BaseDataModel
from src.models.enums import SentimentEnum


class SwapDatasetRecordModel(BaseDataModel):
    sample_id: str
    source_text: str
    target_text: str
    source_sentiment: SentimentEnum
    target_sentiment: SentimentEnum