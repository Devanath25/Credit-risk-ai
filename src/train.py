from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def explain_coefficients(model: LogisticRegression, feature_names: list[str]) -> None:
    """Print coefficient-based feature importance for beginners."""
    coefficients = model.coef_[0]
    ranking = sorted(
        zip(feature_names, coefficients),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    print("\nFeature influence (higher absolute value = stronger effect):")
    for feature, coef in ranking:
        direction = "increases risk" if coef > 0 else "decreases risk"
        print(f"- {feature}: {coef:.3f} ({direction})")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "dataset.csv"
    model_path = project_root / "model.pkl"

    df = pd.read_csv(data_path)

    feature_columns = [
        "income",
        "credit_utilization",
        "missed_payments",
        "debt_to_income",
    ]
    target_column = "delinquent"

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    explain_coefficients(model, feature_columns)

    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
