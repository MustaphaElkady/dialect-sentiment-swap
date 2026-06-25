import json
from pathlib import Path

from src.core import get_settings
from src.helpers import project_path
from src.models import TrainingExampleModel
from src.services.TrainingDataService import TrainingDataService


class LlamaFactoryDataExportService:
    """
    Export project training examples into LLaMA-Factory SFT format.
    """

    def __init__(self):
        self.settings = get_settings()
        self.training_data_service = TrainingDataService()

    def export(self) -> dict[str, Path]:
        train_examples = self.training_data_service.load_train_examples()
        eval_examples = self.training_data_service.load_eval_examples()

        output_dir = project_path(self.settings.LLAMAFACTORY_DATA_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / self.settings.LLAMAFACTORY_TRAIN_FILE
        eval_path = output_dir / self.settings.LLAMAFACTORY_EVAL_FILE
        dataset_info_path = output_dir / self.settings.LLAMAFACTORY_DATASET_INFO_FILE

        train_records = [
            self._build_sft_record(example)
            for example in train_examples
        ]

        eval_records = [
            self._build_sft_record(example)
            for example in eval_examples
        ]

        dataset_info = self._build_dataset_info()

        self._save_json_list(train_records, train_path)
        self._save_json_list(eval_records, eval_path)
        self._save_json_object(dataset_info, dataset_info_path)

        return {
            "train_path": train_path,
            "eval_path": eval_path,
            "dataset_info_path": dataset_info_path,
        }

    def _build_sft_record(self, example: TrainingExampleModel) -> dict:
        source_sentiment = self._sentiment_label(example.source_sentiment)
        target_sentiment = self._sentiment_label(example.target_sentiment)

        system_message = (
            "You are an Arabic sentiment rewriting assistant. "
            "Rewrite Arabic and dialectal Arabic sentences while preserving "
            "meaning, topic, dialect, and style. "
            "Return only the rewritten sentence."
        )

        instruction = "\n".join(
            [
                "Rewrite the following Arabic sentence by changing its sentiment.",
                "",
                f"Current sentiment: {source_sentiment}",
                f"Target sentiment: {target_sentiment}",
                "",
                "Rules:",
                "- Preserve the main meaning, topic, dialect, and writing style.",
                "- Change only sentiment-bearing words or phrases.",
                "- Use suitable emojis only when natural and appropriate.",
                "- Do not explain.",
                "- Do not repeat the instruction.",
                "- Output only the rewritten sentence.",
                "",
                "Original sentence:",
                example.source_text,
                "",
                "Rewritten sentence:",
            ]
        )

        return {
            "system": system_message,
            "instruction": instruction,
            "input": "",
            "output": example.target_text,
            "history": [],
        }

    def _build_dataset_info(self) -> dict:
        return {
            "sentiment_swap_train": {
                "file_name": self.settings.LLAMAFACTORY_TRAIN_FILE,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system",
                    "history": "history",
                },
            },
            "sentiment_swap_val": {
                "file_name": self.settings.LLAMAFACTORY_EVAL_FILE,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system",
                    "history": "history",
                },
            },
        }

    def _sentiment_label(self, sentiment: str) -> str:
        labels = {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
        }

        return labels.get(sentiment, sentiment)

    def _save_json_list(self, records: list[dict], file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(
                records,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def _save_json_object(self, record: dict, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(
                record,
                file,
                ensure_ascii=False,
                indent=2,
            )