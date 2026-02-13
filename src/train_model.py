import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_model():
    # Load training data
    df = pd.read_csv("../data/train.csv")

    # Separate features and target
    X = df.drop("income", axis=1)
    y = df["income"]

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # Evaluate
    print("Validation Accuracy:", model.score(X_val, y_val))

    # Save model
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/income_model.pkl")
    print("Model saved at ../models/income_model.pkl")

if __name__ == "__main__":
    train_model()
