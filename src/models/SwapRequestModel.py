from typing import Optional
from pydantic import Field

from .BaseDataModel import BaseDataModel
from .enums.SentimentEnum import SentimentEnum


class SwapRequestModel(BaseDataModel):
    source_text: str = Field(
        ...,
        min_length=5,
        max_length=512,
        description="Original Arabic or dialectal Arabic sentence that should be rewritten."
    )

    target_sentiment: SentimentEnum = Field(
        ...,
        description="Desired sentiment of the rewritten sentence. Allowed values are: negative, neutral, positive."
    )

    dialect: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=30,
        description="Optional Arabic dialect label for the source sentence, such as Egyptian, Gulf, Levantine, Iraqi, Moroccan, or MSA."
    )

    sample_id: Optional[str] = Field(
        default=None,
        description="Optional unique identifier used to track the sample during prediction, evaluation, or submission."
    )