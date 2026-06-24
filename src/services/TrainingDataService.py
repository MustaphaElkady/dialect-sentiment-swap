import json
from pathlib import Path
from typing import Any

from src.core.config import get_settings
from src.helpers import processed_data_path, load_jsonl
from src.models import TrainingExampleModel

class TrainingDataService:
    def __init__(self):
        self.settings = get_settings()

    def load_train_examples(self) -> list[TrainingExampleModel]:
        train_path = processed_data_path(self.settings.PROCESSED_TRAIN_FILE)
        return self._load_examples(train_path)


    def load_eval_examples(self) -> list[TrainingExampleModel]:
        eval_path = processed_data_path(self.settings.PROCESSED_EVAL_FILE)
        return self._load_examples(eval_path)

    def _load_examples(self, file_path: Path) ->list[TrainingExampleModel]:
        records = load_jsonl(file_path) 

        examples: list[TrainingExampleModel] = []

        for _, record in enumerate(records):
            sample_id = str(record['sample_id'])
            source_text = str(record['source_text'])
            target_text = str(record['target_text'])
            source_sentiment = str(record['source_sentiment']) 
            target_sentiment = str(record['target_sentiment'])

            input_text = self._build_input_text(
                            source_text=source_text,
                            target_sentiment=target_sentiment,
                            source_sentiment=source_sentiment)

            examples.append(TrainingExampleModel(
                sample_id=sample_id,
                input_text=input_text,
                target_text=target_text,
                source_text=source_text,
                target_sentiment=target_sentiment,
            ))      
        return examples

    def _build_input_text(self, source_text: str, target_sentiment: str, source_sentiment: str) -> str:
        return "\n".join(
            [
                "Task: Arabic sentiment rewriting.",
                "Read the sentence, infer its dialect internally, and identify the sentiment-bearing words internally.",
                f"Rewrite the sentence from {source_sentiment} to {target_sentiment}.",
                "Change only the words or phrases needed to match the target sentiment.",
                "Preserve the general meaning, topic, style, and dialect as much as possible.",
                "Output only the rewritten sentence.",
                "",
                f"Sentence: {source_text}",
                "Rewritten sentence:",
            ]
        )