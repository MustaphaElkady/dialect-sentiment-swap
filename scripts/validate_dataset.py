from src.services.DatasetValidationService import DatasetValidationService


def main() -> None:
    service = DatasetValidationService()
    service.validate()


if __name__ == "__main__":
    main()