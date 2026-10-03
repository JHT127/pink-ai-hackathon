"""Read raw project data without converting or cleaning cell values."""

import pandas as pd

from src.config import TEST_RAW, TRAIN_RAW


def load_raw(path):
    """Read a raw CSV as text and strip whitespace from column names only."""
    # Keep cell values as they appear in the source file.
    data = pd.read_csv(path, header=1, dtype=str, keep_default_na=False)
    data.columns = data.columns.str.strip()
    return data


def load_train():
    """Load the raw training CSV without changing cell values."""
    # Use the configured raw training path.
    return load_raw(TRAIN_RAW)


def load_test():
    """Load the raw test CSV without changing cell values."""
    # Use the configured raw test path.
    return load_raw(TEST_RAW)