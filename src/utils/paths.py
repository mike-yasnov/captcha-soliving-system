from pathlib import Path
import os


def get_project_path() -> str:
    """Get the project path 

    Returns:
        str: path to project
    """

    path = Path(__file__).parent.parent.parent

    return path


PATH_TO_YOLO_MODEL = os.path.join(
    get_project_path(), "weights", "best.pt"
)
