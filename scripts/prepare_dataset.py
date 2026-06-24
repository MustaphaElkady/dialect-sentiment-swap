from src.services.DataPreparationService import DataPreparationService


def main() -> None:
    service = DataPreparationService()
    service.prepare()


if __name__ == "__main__":
    main()