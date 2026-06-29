from typing import Optional
from pydantic import Field

from .BaseDataModel import BaseDataModel
from .enums.SentimentEnum import SentimentEnum


class SwapResultModel(BaseDataModel):
    source_text: str = Field(
        ...,
        description="Original Arabic or dialectal Arabic sentence that was provided as input."
    )

    target_sentiment: SentimentEnum = Field(
        ...,
        description="Target sentiment requested for the rewritten sentence. Allowed values are: negative, neutral, positive."
    )

    generated_text: str = Field(
        ...,
        description="Generated rewritten sentence after changing the sentiment while preserving meaning, topic, and dialect style."
    )

    success: bool = Field(
        default=False,
        description="Indicates whether the sentiment rewriting process completed successfully."
    )

    dialect: Optional[str] = Field(
        default=None,
        description="Optional Arabic dialect label detected or provided for the sentence, such as Egyptian, Gulf, Levantine, Iraqi, Moroccan, or MSA."
    )

    sample_id: Optional[str] = Field(
        default=None,
        description="Optional unique identifier used to track the sample during prediction, evaluation, or submission."
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Optional error message returned when the generation process fails."
    )