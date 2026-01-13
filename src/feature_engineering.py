import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_features(
    data_path: str,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Load cleaned data, encode categorical features,
    split into train and test sets.
    """

    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # One-hot encoding for categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
