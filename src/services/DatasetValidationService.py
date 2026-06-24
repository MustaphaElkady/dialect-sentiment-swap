import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from src.core.config import get_settings
from src.helpers import load_jsonl


class DatasetValidationService:
    """
    Validate processed JSONL datasets before training and save a report file.
    """

    REQUIRED_FIELDS = {
        "sample_id",
        "source_text",
        "target_text",
        "source_sentiment",
        "target_sentiment",
    }

    def __init__(self):
        self.settings = get_settings()

    def validate(self) -> None:
        train_path = self._processed_path(self.settings.PROCESSED_TRAIN_FILE)
        eval_path = self._processed_path(self.settings.PROCESSED_EVAL_FILE)
        report_path = self._processed_path(
            self.settings.DATASET_VALIDATION_REPORT_FILE
        )

        report_lines: list[str] = []

        report_lines.extend(self._validate_file("Train", train_path))
        report_lines.append("")
        report_lines.extend(self._validate_file("Eval", eval_path))

        report_text = "\n".join(report_lines)

        print(report_text)

        self._save_report(report_text, report_path)
        print()
        print(f"Validation report saved to: {report_path}")

    def _validate_file(self, split_name: str, file_path: Path) -> list[str]:
        if not file_path.exists():
            raise FileNotFoundError(f"{split_name} file not found: {file_path}")

        records = self.load_jsonl(file_path)

        lines: list[str] = []

        lines.append("=" * 70)
        lines.append(f"{split_name} Dataset Report")
        lines.append("=" * 70)
        lines.append(f"File path: {file_path}")
        lines.append(f"Total records: {len(records)}")
        lines.append("")

        lines.extend(self._get_missing_fields_report(records))
        lines.extend(self._get_empty_values_report(records))
        lines.extend(self._get_sentiment_distribution(records))
        lines.extend(self._get_same_text_report(records))
        lines.extend(self._get_duplicate_ids_report(records))
        lines.extend(self._get_random_examples(records))

        return lines

    def _get_missing_fields_report(
        self,
        records: list[dict[str, Any]],
    ) -> list[str]:
        rows_with_missing_fields = 0

        for record in records:
            missing_fields = self.REQUIRED_FIELDS - set(record.keys())

            if missing_fields:
                rows_with_missing_fields += 1

        return [
            f"Rows with missing fields: {rows_with_missing_fields}",
        ]

    def _get_empty_values_report(
        self,
        records: list[dict[str, Any]],
    ) -> list[str]:
        empty_source_count = 0
        empty_target_count = 0

        for record in records:
            if not str(record.get("source_text", "")).strip():
                empty_source_count += 1

            if not str(record.get("target_text", "")).strip():
                empty_target_count += 1

        return [
            f"Empty source_text rows: {empty_source_count}",
            f"Empty target_text rows: {empty_target_count}",
        ]

    def _get_sentiment_distribution(
        self,
        records: list[dict[str, Any]],
    ) -> list[str]:
        lines: list[str] = []

        source_counter = Counter(
            record.get("source_sentiment")
            for record in records
        )

        target_counter = Counter(
            record.get("target_sentiment")
            for record in records
        )

        lines.append("Source sentiment distribution:")
        for sentiment, count in sorted(source_counter.items()):
            lines.append(f"  - {sentiment}: {count}")

        lines.append("Target sentiment distribution:")
        for sentiment, count in sorted(target_counter.items()):
            lines.append(f"  - {sentiment}: {count}")

        return lines

    def _get_same_text_report(
        self,
        records: list[dict[str, Any]],
    ) -> list[str]:
        same_text_count = 0

        for record in records:
            source_text = str(record.get("source_text", "")).strip()
            target_text = str(record.get("target_text", "")).strip()

            if source_text == target_text:
                same_text_count += 1

        return [
            f"Rows where source_text equals target_text: {same_text_count}",
        ]

    def _get_duplicate_ids_report(
        self,
        records: list[dict[str, Any]],
    ) -> list[str]:
        sample_ids = [
            record.get("sample_id")
            for record in records
        ]

        duplicate_count = len(sample_ids) - len(set(sample_ids))

        return [
            f"Duplicate sample_id rows: {duplicate_count}",
        ]

    def _get_random_examples(
        self,
        records: list[dict[str, Any]],
    ) -> list[str]:
        lines: list[str] = []

        lines.append("")
        lines.append("Random examples:")

        if not records:
            lines.append("  No records available.")
            return lines

        sample_size = min(3, len(records))
        examples = random.sample(records, sample_size)

        for index, record in enumerate(examples, start=1):
            lines.append(f"  Example {index}:")
            lines.append(f"    source_sentiment: {record.get('source_sentiment')}")
            lines.append(f"    target_sentiment: {record.get('target_sentiment')}")
            lines.append(f"    source_text: {record.get('source_text')}")
            lines.append(f"    target_text: {record.get('target_text')}")

        return lines

    def _save_report(self, report_text: str, report_path: Path) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as file:
            file.write(report_text)
            file.write("\n")
