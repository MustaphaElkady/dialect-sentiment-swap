from pathlib import Path
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core import get_settings
from src.helpers import project_path, save_jsonl
from src.models import GenerationPredictionModel, TrainingExampleModel,SwapResultModel


class CausalGenerationService:
    """
    Run baseline generation using an instruction-tuned causal language model.
    """

    def __init__(self):
        self.settings = get_settings()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.CAUSAL_MODEL_NAME,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.CAUSAL_MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        self.model.eval()

    def generate_predictions(
        self,
        examples: list[TrainingExampleModel],
    ) -> list[GenerationPredictionModel]:
        selected_examples = examples[: self.settings.CAUSAL_MAX_EXAMPLES]

        predictions: list[GenerationPredictionModel] = []

        for index, example in enumerate(selected_examples, start=1):
            print(
                f"Generating {index}/{len(selected_examples)} "
                f"for sample_id={example.sample_id}"
            )

            prompt = self._build_prompt(example)
            prediction_text = self._generate_text(prompt)

            prediction = GenerationPredictionModel(
                sample_id=example.sample_id,
                model_name=self.settings.CAUSAL_MODEL_NAME,
                input_text=prompt,
                source_text=example.source_text,
                reference_text=example.target_text,
                prediction_text=prediction_text,
                target_sentiment=example.target_sentiment,
            )

            predictions.append(prediction)

        return predictions

    def save_predictions(
        self,
        predictions: list[GenerationPredictionModel],
    ) -> Path:
        output_path = self._causal_output_path()

        prediction_records = [
            prediction.model_dump(mode="json")
            for prediction in predictions
        ]

        save_jsonl(prediction_records, output_path)

        return output_path

    def _generate_text(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "أنت مساعد متخصص في إعادة كتابة الجمل العربية واللهجات العربية.",
                        "مهمتك هي تغيير الشعور فقط مع الحفاظ على المعنى العام والموضوع واللهجة والأسلوب.",
                        "لا تغيّر أسماء الأشخاص أو الأماكن أو المنتجات.",
                        "لا تضف معلومات جديدة غير موجودة في الجملة الأصلية.",
                        "لا تشرح.",
                        "لا تكرر التعليمات.",
                        "اكتب الجملة الناتجة فقط.",
                    ]
                ),
            },
            {
                "role": "user",
                "content":"\n".join([
                    prompt,
                    "",
                    "### Pydantic details: ",
                    json.dumps(
                    SwapResultModel.moedl_json_schema(),ensure_ascii=False
                    )
                    ]) 
            },
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        encoded_input = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=getattr(self.settings, "CAUSAL_MAX_INPUT_TOKENS", 1024),
        )

        encoded_input = {
            key: value.to(self.model.device)
            for key, value in encoded_input.items()
        }

        input_length = encoded_input["input_ids"].shape[-1]

        with torch.no_grad():
            generated_ids = self.model.generate(
                **encoded_input,
                max_new_tokens=self.settings.CAUSAL_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = generated_ids[0][input_length:]

        prediction_text = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        return self._clean_prediction(prediction_text)

    def _build_prompt(self, example: TrainingExampleModel) -> str:
        sentiment_labels = {
            "positive": "إيجابي (positive)",
            "negative": "سلبي (negative)",
            "neutral": "محايد (neutral)",
        }

        source_sentiment = sentiment_labels.get(
            example.source_sentiment,
            example.source_sentiment,
        )

        target_sentiment = sentiment_labels.get(
            example.target_sentiment,
            example.target_sentiment,
        )

        return "\n".join(
            [
                "أعد كتابة الجملة العربية التالية مع تغيير الشعور فقط.",
                "",
                f"### الشعور الحالي: {source_sentiment} ",
                f"### الشعور المطلوب: {target_sentiment}",
                "",
                "### القواعد:",
                "- حافظ على نفس المعنى العام.",
                "- حافظ على نفس الموضوع والوصف.",
                "- حافظ على اللهجة والأسلوب قدر الإمكان.",
                "- غيّر فقط الكلمات أو العبارات المرتبطة بالشعور.",
                "- لا تضف شرحًا أو ملاحظات.",
                "- اكتب الجملة الناتجة فقط.",
                "",
                f"### الجملة الأصلية: {example.source_text}",
                "",
                "### الجملة الناتجة:",
            ]
        )

    def _clean_prediction(self, text: str) -> str:
        text = text.strip()

        prefixes = [
            "الجملة الناتجة:",
            "الناتج:",
            "الإجابة:",
            "الإجابة النهائية:",
            "Output:",
            "Result:",
        ]

        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        return text

    def _causal_output_path(self) -> Path:
        return (
            project_path(self.settings.CAUSAL_OUTPUT_DIR)
            / self.settings.CAUSAL_PREDICTIONS_FILE
        )