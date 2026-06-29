import json
import random
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

        random.Random(101).shuffle(train_records)

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

        system_message = "\n".join(
    [
        "أنت مساعد متخصص في إعادة كتابة الجمل العربية واللهجات العربية.",
        "مهمتك هي تغيير الشعور فقط مع الحفاظ على المعنى العام والموضوع والوصف واللهجة والأسلوب.",
        "غيّر فقط الكلمات أو العبارات المرتبطة بالشعور.",
        "لا تغيّر أسماء الأشخاص أو الأماكن أو المنتجات.",
        "لا تضف معلومات جديدة غير موجودة في الجملة الأصلية.",

        "التزم بالاحترام الديني الكامل.",
        "لا تسيء إلى الله أو الأنبياء أو القرآن أو الدين أو الشعائر أو الدعاء.",
        "لا تستبدل الألفاظ الدينية مثل: الله، الرحمن، القرآن، الدعاء، الجنة، الصلاة، الحمد لله، ما شاء الله، إن شاء الله، بأي ألفاظ سلبية أو مسيئة.",
        "إذا كانت الجملة تحتوي على دعاء أو تهنئة دينية، حافظ على الألفاظ الدينية كما هي.",
        "عند تحويل جملة دينية أو دعاء من إيجابي إلى سلبي، استخدم صياغة آمنة ومحترمة بدون إساءة دينية.",

        "لا تشرح.",
        "لا تكرر التعليمات.",
        "اكتب الجملة الناتجة فقط.",
    ]
)

        instruction = "\n".join(
            [
                "أعد كتابة الجملة العربية التالية مع تغيير الشعور فقط.",
                "",
                f"### الشعور الحالي: {source_sentiment}",
                f"### الشعور المطلوب: {target_sentiment}",
                "",
                "### القواعد:",
                "- حافظ على نفس المعنى العام.",
                "- حافظ على نفس الموضوع والوصف.",
                "- حافظ على اللهجة والأسلوب قدر الإمكان.",
                "- غيّر فقط الكلمات أو العبارات المرتبطة بالشعور.",
                "- لا تغيّر الألفاظ الدينية أو الدعاء أو أسماء الله أو القرآن أو الصلاة.",
                "- لا تنتج أي إساءة دينية أو سب أو كفر أو ألفاظ مخالفة للشريعة الإسلامية.",
                "- إذا كانت الجملة تحتوي على دعاء أو تهنئة دينية، حافظ عليها بصياغة محترمة وآمنة.",
                "- استخدم رموزًا تعبيرية مناسبة إذا كانت موجودة أو مناسبة للسياق.",
                "- لا تضف شرحًا أو ملاحظات.",
                "- اكتب الجملة الناتجة فقط.",
                f"### الجملة الأصلية: {example.source_text}",
                "",
                "### الجملة الناتجة:",
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
        sentiment_value = getattr(sentiment, "value", sentiment)

        labels = {
            "positive": "إيجابي (positive)",
            "negative": "سلبي (negative)",
            "neutral": "محايد (neutral)",
        }

        return labels.get(sentiment_value, sentiment_value)

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