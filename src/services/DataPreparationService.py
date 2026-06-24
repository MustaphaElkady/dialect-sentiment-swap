import json
from pathlib import Path

import pandas as pd

from src.core.config import get_settings
from src.models import SwapDatasetRecordModel
from src.models.enums import SentimentEnum


class DataPreparationService:
    """
    Prepare raw Excel sentiment swap data into normalized JSONL files.
    """

    REQUIRED_COLUMNS = {
        "id",
        "source",
        "source_polarity",
        "target",
    }

    def __init__(self):
        self.settings = get_settings()

    def prepare(self) -> None:
        train_input_path = self._raw_path(self.settings.RAW_TRAIN_FILE)
        eval_input_path = self._raw_path(self.settings.RAW_EVAL_FILE)

        train_output_path = self._processed_path(self.settings.PROCESSED_TRAIN_FILE)
        eval_output_path = self._processed_path(self.settings.PROCESSED_EVAL_FILE)

        train_records = self._load_excel_records(train_input_path)
        eval_records = self._load_excel_records(eval_input_path)

        self._save_jsonl(train_records, train_output_path)
        self._save_jsonl(eval_records, eval_output_path)

        print(f"Train records: {len(train_records)}")
        print(f"Eval records: {len(eval_records)}")
        print(f"Saved train to: {train_output_path}")
        print(f"Saved eval to: {eval_output_path}")

    def _load_excel_records(self, file_path: Path) -> list[SwapDatasetRecordModel]:
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")

        df = pd.read_excel(file_path)
        self._validate_columns(df=df, file_path=file_path)

        records: list[SwapDatasetRecordModel] = []

        for _, row in df.iterrows():
            source_text = str(row["source"]).strip()
            target_text = str(row["target"]).strip()

            if not source_text or not target_text:
                continue

            if source_text == target_text:
                continue

            source_sentiment = self._normalize_sentiment(row["source_polarity"])
            target_sentiment = self._get_opposite_sentiment(source_sentiment)

            record = SwapDatasetRecordModel(
                sample_id=str(row["id"]),
                source_text=source_text,
                target_text=target_text,
                source_sentiment=source_sentiment,
                target_sentiment=target_sentiment,
            )

            records.append(record)

        return records

    def _validate_columns(self, df: pd.DataFrame, file_path: Path) -> None:
        missing_columns = self.REQUIRED_COLUMNS - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns in {file_path}: {sorted(missing_columns)}"
            )

    def _normalize_sentiment(self, value: str) -> SentimentEnum:
        normalized = str(value).strip().lower()

        if normalized == "positive":
            return SentimentEnum.POSITIVE

        if normalized == "negative":
            return SentimentEnum.NEGATIVE

        if normalized == "neutral":
            return SentimentEnum.NEUTRAL

        raise ValueError(f"Unsupported sentiment value: {value}")

    def _get_opposite_sentiment(self, sentiment: SentimentEnum) -> SentimentEnum:
        if sentiment == SentimentEnum.POSITIVE:
            return SentimentEnum.NEGATIVE

        if sentiment == SentimentEnum.NEGATIVE:
            return SentimentEnum.POSITIVE

        return SentimentEnum.NEUTRAL

    def _save_jsonl(
        self,
        records: list[SwapDatasetRecordModel],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            for record in records:
                file.write(
                    json.dumps(
                        record.model_dump(mode="json"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _raw_path(self, file_name: str) -> Path:
        return self._project_path(self.settings.RAW_DATA_DIR) / file_name

    def _processed_path(self, file_name: str) -> Path:
        return self._project_path(self.settings.PROCESSED_DATA_DIR) / file_name

    def _project_path(self, path: str) -> Path:
        path_obj = Path(path)

        if path_obj.is_absolute():
            return path_obj

        return Path(self.settings.PROJECT_ROOT) / path_obj