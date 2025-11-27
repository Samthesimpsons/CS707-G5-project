from pathlib import Path

# Base directories
PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent

# Data/model locations in the current repo layout
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
QA_OUTPUT_DIR = DATA_DIR / "qa_output"
MODELS_DIR = PROJECT_ROOT / "models"

# Goldfish-specific assets
CHECKPOINTS_DIR = PACKAGE_ROOT / "checkpoints"
TEST_CONFIGS_DIR = PACKAGE_ROOT / "test_configs"
WORKSPACE_DIR = PACKAGE_ROOT / "new_workspace"
LEGACY_WORKSPACE_DIR = PACKAGE_ROOT / "workspace"
RESULTS_DIR = PROJECT_ROOT / "results_goldfish"


def ensure_dir(path: Path) -> Path:
    """Create directory if missing and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
