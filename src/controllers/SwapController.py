from src.models import SwapRequestModel, SwapResultModel
from src.services.RuleBasedSwapService import RuleBasedSwapService


class SwapController:
    """
    Coordinates the sentiment swap pipeline.

    The controller receives a structured request,
    calls the service that performs the generation,
    then returns a structured result.
    """

    def __init__(self, swap_service: RuleBasedSwapService | None = None):
        self.swap_service = swap_service or RuleBasedSwapService()

    def swap(self, request: SwapRequestModel) -> SwapResultModel:
        try:
            generated_text = self.swap_service.generate(request)

            success = generated_text != request.source_text

            return SwapResultModel(
                source_text=request.source_text,
                target_sentiment=request.target_sentiment,
                generated_text=generated_text,
                success=success,
                dialect=request.dialect,
                sample_id=request.sample_id,
            )

        except Exception as error:
            return SwapResultModel(
                source_text=request.source_text,
                target_sentiment=request.target_sentiment,
                generated_text="",
                success=False,
                dialect=request.dialect,
                sample_id=request.sample_id,
                error_message=str(error),
            )