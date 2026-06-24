from scipy._lib.array_api_compat import device
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.core import get_settings
from src.helpers import project_path, save_jsonl
from src.models import GenerationPredictionModel, TrainingExampleModel

class BaselineGenerationService:
    """
    Run baseline generation using a pretrained Seq2Seq model before fine-tuning.
    """

    def __init__(self):
        self.settings = get_settings()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.TRAIN_MODEL_NAME,
            use_fast=False,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.settings.TRAIN_MODEL_NAME
        )
        self.model.to(self.device)
        self.model.eval()
    def generate_predictions(
        self,
        examples: list[TrainingExampleModel],
    ) -> list[GenerationPredictionModel]:
        selected_examples = examples[: self.settings.BASELINE_MAX_EXAMPLES]

        predictions: list[GenerationPredictionModel] = []

        for index, example in enumerate(selected_examples, start=1):
            print(
                f"Generating {index}/{len(selected_examples)} "
                f"for sample_id={example.sample_id}"
            )

            prediction_text = self._generate_text(example.input_text)

            prediction = GenerationPredictionModel(
                sample_id=example.sample_id,
                model_name=self.settings.TRAIN_MODEL_NAME,
                input_text=example.input_text,
                source_text=example.source_text,
                reference_text=example.target_text,
                prediction_text=prediction_text,
                target_sentiment=example.target_sentiment,
            )

            predictions.append(prediction)

        return predictions

    def _generate_text(self, input_text: str) -> str:
        encoded_input = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.settings.MAX_SOURCE_LENGTH,
            truncation=True,
        )

        encoded_input = {
            key: value.to(self.device)
            for key, value in encoded_input.items()
        }

        with torch.no_grad():
            generated_ids = self.model.generate(
                **encoded_input,
                max_new_tokens=self.settings.GENERATION_MAX_NEW_TOKENS,
                num_beams=self.settings.GENERATION_NUM_BEAMS,
                early_stopping=True,
            )

        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        return generated_text.strip()
    def _generate_text(self, input_text: str) -> str:
        encoded_input = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.settings.MAX_SOURCE_LENGTH,
            truncation=True,
        )

        encoded_input = {
            key: value.to(self.device)
            for key, value in encoded_input.items()
        }

        with torch.no_grad():
            generated_ids = self.model.generate(
                **encoded_input,
                max_new_tokens=self.settings.GENERATION_MAX_NEW_TOKENS,
                num_beams=self.settings.GENERATION_NUM_BEAMS,
                early_stopping=True,
            )

        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        return generated_text.strip()

    def save_predictions(
        self,
        predictions: list[GenerationPredictionModel],
    ) -> Path:
        output_path = self._baseline_output_path()

        prediction_records = [
            prediction.model_dump(mode="json")
            for prediction in predictions
        ]

        save_jsonl(prediction_records, output_path)

        return output_path

    def _baseline_output_path(self) -> Path:
        return (
            project_path(self.settings.BASELINE_OUTPUT_DIR)
            / self.settings.BASELINE_PREDICTIONS_FILE
        )