from typing import Optional

from .BaseDataModel import BaseDataModel
from .enums.SentimentEnum import SentimentEnum

class SwapResultModel(BaseDataModel):
    source_text: str
    target_sentiment : SentimentEnum
    generated_text: str
    success: bool = False
    dialect: Optional[str] = None
    sample_id: Optional[str] = None
    error_message: Optional[str] = None