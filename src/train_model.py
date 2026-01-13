from sklearn.linear_model import LogisticRegression
from feature_engineering import prepare_features


DATA_PATH = "../data/processed/cleaned_data.csv"
TARGET_COLUMN = "Survived"  # change if your dataset has a different target


def train():
    X_train, X_test, y_train, y_test = prepare_features(
        DATA_PATH, TARGET_COLUMN
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_test, y_test


if __name__ == "__main__":
    train()
