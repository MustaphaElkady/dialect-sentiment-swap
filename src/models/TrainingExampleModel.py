from src.models.BaseDataModel import BaseDataModel


class TrainingExampleModel(BaseDataModel):
    sample_id: str
    input_text: str
    target_text: str
    source_text: str
    target_sentiment: str