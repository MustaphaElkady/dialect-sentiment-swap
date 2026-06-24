from typing import Optional

from .BaseDataModel import BaseDataModel
from .enums.SentimentEnum import SentimentEnum

class SwapRequestModel(BaseDataModel):
    source_text: str
    target_sentiment : SentimentEnum
    dialect: Optional[str] = None
    sample_id: Optional[str] = None