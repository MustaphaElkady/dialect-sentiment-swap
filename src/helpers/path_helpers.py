from pathlib import Path
from src.core.config import get_settings

def project_path(path: Path | str) -> Path:
    settings = get_settings()
    
    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj

    return Path(settings.PROJECT_ROOT) / path_obj

def raw_data_path(file_name: str|Path) -> Path:
    settings = get_settings()
    return project_path(
        Path(settings.RAW_DATA_DIR) / file_name
    )

def processed_data_path(file_name: str | Path) -> Path:
    settings = get_settings()

    return project_path(
        Path(settings.PROCESSED_DATA_DIR) / file_name
    )