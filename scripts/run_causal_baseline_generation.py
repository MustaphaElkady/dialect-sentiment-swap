from src.services.CausalGenerationService import CausalGenerationService
from src.services.TrainingDataService import TrainingDataService


def main() -> None:
    data_service = TrainingDataService()
    eval_examples = data_service.load_eval_examples()

    generation_service = CausalGenerationService()
    predictions = generation_service.generate_predictions(eval_examples)

    output_path = generation_service.save_predictions(predictions)

    print()
    print(f"Saved causal baseline predictions to: {output_path}")


if __name__ == "__main__":
    main()+