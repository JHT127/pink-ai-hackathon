from pathlib import Path


def repo_root():
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


RAW_DIR = repo_root() / "data" / "raw"
CLEAN_DIR = repo_root() / "data" / "clean"
SUBMISSIONS_DIR = repo_root() / "submissions"
LOGS_DIR = repo_root() / "logs"
TRAIN_RAW = RAW_DIR / "pinkai_train.csv"
TEST_RAW = RAW_DIR / "pinkai_test.csv"
SAMPLE_SUBMISSION = RAW_DIR / "sample_submission.csv"
RANDOM_SEED = 42
TARGET = "diagnosis"
ID_COL = "patient_id"
