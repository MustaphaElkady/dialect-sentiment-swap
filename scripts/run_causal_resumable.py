import json
import os
from pathlib import Path

from src.helpers import project_path
from src.models import GenerationPredictionModel
from src.services.CausalGenerationService import CausalGenerationService
from src.services.TrainingDataService import TrainingDataService


def load_done_ids(output_path: Path) -> set[str]:
    done_ids = set()

    if not output_path.exists():
        return done_ids

    with open(output_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            sample_id = str(record.get("sample_id", "")).strip()

            if sample_id:
                done_ids.add(sample_id)

    return done_ids


def append_jsonl(output_path: Path, record: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    start = int(os.getenv("START", "0"))
    size = int(os.getenv("SIZE", "100"))

    data_service = TrainingDataService()
    eval_examples = data_service.load_eval_examples()

    examples = eval_examples[start : start + size]

    generation_service = CausalGenerationService()

    output_path = (
        project_path(generation_service.settings.CAUSAL_OUTPUT_DIR)
        / generation_service.settings.CAUSAL_PREDICTIONS_FILE
    )

    done_ids = load_done_ids(output_path)

    print(f"Total eval: {len(eval_examples)}")
    print(f"Start: {start}")
    print(f"Size: {size}")
    print(f"Selected: {len(examples)}")
    print(f"Already done: {len(done_ids)}")
    print(f"Output: {output_path}")
    print()

    for index, example in enumerate(examples, start=1):
        if example.sample_id in done_ids:
            print(f"Skip {index}/{len(examples)} | sample_id={example.sample_id}")
            continue

        print(f"Generate {index}/{len(examples)} | sample_id={example.sample_id}")

        prompt = generation_service._build_prompt(example)
        prediction_text = generation_service._generate_text(prompt)

        prediction = GenerationPredictionModel(
            sample_id=example.sample_id,
            model_name=generation_service.settings.CAUSAL_MODEL_NAME,
            input_text=prompt,
            source_text=example.source_text,
            reference_text=example.target_text,
            prediction_text=prediction_text,
            target_sentiment=example.target_sentiment,
        )

        append_jsonl(
            output_path=output_path,
            record=prediction.model_dump(mode="json"),
        )

        done_ids.add(example.sample_id)

    print()
    print("Done.")


if __name__ == "__main__":
    main()