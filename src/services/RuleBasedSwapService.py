from src.models import SwapRequestModel
from src.models.enums import SentimentEnum


class RuleBasedSwapService:
    """
    Simple first version of the swap logic.

    This is not the final AI model.
    It only proves that the project layers can communicate correctly.
    """

    def generate(self, request: SwapRequestModel) -> str:
        source_text = request.source_text

        if request.target_sentiment == SentimentEnum.POSITIVE:
            return self._to_positive(source_text)

        if request.target_sentiment == SentimentEnum.NEGATIVE:
            return self._to_negative(source_text)

        return self._to_neutral(source_text)

    def _to_positive(self, text: str) -> str:
        replacements = {
            "سيئة": "ممتازة",
            "سيء": "ممتاز",
            "وحشة": "جميلة",
            "رديئة": "رائعة",
            "مشكلة": "ميزة",
        }

        return self._replace_words(text, replacements)

    def _to_negative(self, text: str) -> str:
        replacements = {
            "ممتازة": "سيئة",
            "ممتاز": "سيء",
            "جميلة": "وحشة",
            "رائعة": "رديئة",
            "ميزة": "مشكلة",
        }

        return self._replace_words(text, replacements)

    def _to_neutral(self, text: str) -> str:
        replacements = {
            "ممتازة": "عادية",
            "ممتاز": "عادي",
            "سيئة": "عادية",
            "سيء": "عادي",
            "رائعة": "عادية",
            "رديئة": "عادية",
            "جميلة": "عادية",
            "وحشة": "عادية",
        }

        return self._replace_words(text, replacements)

    def _replace_words(self, text: str, replacements: dict[str, str]) -> str:
        generated_text = text

        for old_word, new_word in replacements.items():
            generated_text = generated_text.replace(old_word, new_word)

        return generated_text