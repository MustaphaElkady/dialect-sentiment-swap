from src.models.BaseDataModel import BaseDataModel


class GenerationPredictionModel(BaseDataModel):
    sample_id: str
    model_name: str
    input_text: str
    source_text: str
    reference_text: str
    prediction_text: str
    target_sentiment: str