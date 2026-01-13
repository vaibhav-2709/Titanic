import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - Numerical columns: median
    - Categorical columns: mode
    """
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop irrelevant or high-missing-value columns (if present).
    """
    columns_to_drop = ["Cabin", "Ticket", "Name", "Id"]

    existing_columns = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=existing_columns)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform complete data cleaning pipeline.
    """
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = drop_unnecessary_columns(df)

    return df
