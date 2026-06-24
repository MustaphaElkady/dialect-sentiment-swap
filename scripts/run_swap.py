from src.controllers.SwapController import SwapController
from src.models import SwapRequestModel
from src.models.enums import SentimentEnum


def main() -> None:
    request = SwapRequestModel(
        source_text="الخدمة كانت سيئة جدًا",
        target_sentiment=SentimentEnum.POSITIVE,
        dialect="Egyptian",
        sample_id="demo-001",
    )

    controller = SwapController()
    result = controller.swap(request)

    print("Source:", result.source_text)
    print("Target sentiment:", result.target_sentiment.value)
    print("Generated:", result.generated_text)
    print("Success:", result.success)

    if result.error_message:
        print("Error:", result.error_message)


if __name__ == "__main__":
    main()