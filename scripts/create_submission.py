import json
import zipfile
from pathlib import Path

import pandas as pd

from src.helpers import project_path


PREDICTIONS_FILE = "experiments/baselines/qwen2_5_7b_instruct_predictions.jsonl"

SUBMISSION_DIR = "submissions"
XLSX_FILE_NAME = "predictions.xlsx"
ZIP_FILE_NAME = "predictions.zip"


def load_prediction_records(file_path: Path) -> list[dict]:
    records: list[dict] = []

    if not file_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {file_path}"
                ) from error

    return records


def build_submission_dataframe(records: list[dict]) -> pd.DataFrame:
    submission_rows: list[dict] = []

    for record in records:
        sample_id = str(record.get("sample_id", "")).strip()
        prediction_text = str(record.get("prediction_text", "")).strip()

        if not sample_id:
            raise ValueError("Found prediction record without sample_id.")

        if not prediction_text:
            prediction_text = "."

        submission_rows.append(
            {
                "id": sample_id,
                "style": prediction_text,
            }
        )

    submission_df = pd.DataFrame(submission_rows)

    if submission_df["id"].duplicated().any():
        duplicated_ids = submission_df[
            submission_df["id"].duplicated()
        ]["id"].tolist()

        raise ValueError(f"Duplicated sample ids found: {duplicated_ids[:10]}")

    return submission_df


def save_submission_files(submission_df: pd.DataFrame) -> tuple[Path, Path]:
    output_dir = project_path(SUBMISSION_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = output_dir / XLSX_FILE_NAME
    zip_path = output_dir / ZIP_FILE_NAME

    submission_df.to_excel(
        xlsx_path,
        index=False,
        engine="openpyxl",
    )

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(
            xlsx_path,
            arcname=XLSX_FILE_NAME,
        )

    return xlsx_path, zip_path


def main() -> None:
    predictions_path = project_path(PREDICTIONS_FILE)

    records = load_prediction_records(predictions_path)
    submission_df = build_submission_dataframe(records)
    xlsx_path, zip_path = save_submission_files(submission_df)

    print("Submission file created successfully.")
    print()
    print(f"Rows: {len(submission_df)}")
    print(f"XLSX: {xlsx_path}")
    print(f"ZIP: {zip_path}")


if __name__ == "__main__":
    main()