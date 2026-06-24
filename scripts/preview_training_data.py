from src.core import get_settings
from src.helpers import project_path, save_jsonl
from src.services.TrainingDataService import TrainingDataService


def print_example(title: str, example) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)

    print("sample_id:")
    print(example.sample_id)
    print()

    print("source_sentiment:")
    print(example.source_sentiment)
    print()

    print("target_sentiment:")
    print(example.target_sentiment)
    print()

    print("input_text:")
    print(example.input_text)
    print()

    print("target_text:")
    print(example.target_text)
    print()


def main() -> None:
    settings = get_settings()
    service = TrainingDataService()

    train_examples = service.load_train_examples()
    eval_examples = service.load_eval_examples()

    print("Train examples:", len(train_examples))
    print("Eval examples:", len(eval_examples))
    print()

    print_example("First train example", train_examples[0])
    print_example("First eval example", eval_examples[0])

    output_path = (
        project_path(settings.TRAINING_PREVIEW_OUTPUT_DIR)
        / settings.TRAINING_PREVIEW_FILE
    )

    preview_records = []

    for example in train_examples[: settings.TRAINING_PREVIEW_MAX_EXAMPLES]:
        record = example.model_dump(mode="json")
        record["split"] = "train"
        preview_records.append(record)

    for example in eval_examples[: settings.TRAINING_PREVIEW_MAX_EXAMPLES]:
        record = example.model_dump(mode="json")
        record["split"] = "eval"
        preview_records.append(record)

    save_jsonl(preview_records, output_path)

    print()
    print(f"Saved preview examples to: {output_path}")


if __name__ == "__main__":
    main()