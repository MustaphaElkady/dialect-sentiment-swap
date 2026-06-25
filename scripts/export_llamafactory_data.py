from src.services.LlamaFactoryDataExportService import LlamaFactoryDataExportService


def main() -> None:
    service = LlamaFactoryDataExportService()
    output_paths = service.export()

    print("LLaMA-Factory data exported successfully.")
    print()
    print(f"Train file: {output_paths['train_path']}")
    print(f"Eval file: {output_paths['eval_path']}")
    print(f"Dataset info file: {output_paths['dataset_info_path']}")


if __name__ == "__main__":
    main()