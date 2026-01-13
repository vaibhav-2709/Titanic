from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from train_model import train


def evaluate():
    model, X_test, y_test = train()

    predictions = model.predict(X_test)

    print("Accuracy Score:")
    print(accuracy_score(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    evaluate()
