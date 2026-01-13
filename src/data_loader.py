import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        file_path (str): Path to CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    return pd.read_csv(file_path)
