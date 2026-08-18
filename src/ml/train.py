"""
Model Training and Comparison

Trains and compares multiple machine learning models
for invoice fraud detection.
"""

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# 
# Load Dataset
# 

def load_dataset():

    print("Loading the ML dataset...")

    df = pd.read_csv(
        "data/processed/ml_dataset.csv"
    )

    return df


# 
# Split Features and Target
# 

def split_features_target(df):

    print("Splitting Features and Target...")

    X = df.drop(
        columns=[
            "fraud",
            "risk_score",
            "risk_severity"
        ]
    )

    y = df["fraud"]

    return X, y


# 
# Train/Test Split
# 

def split_dataset(X, y):

    print("Creating Train/Test Split...")

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )

    return X_train, X_test, y_train, y_test


# 
# Compare Multiple Models
# 

def compare_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\nComparing Models...")

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

    }

    results = []

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions
        )

        recall = recall_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        results.append({

            "Model": name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1": f1

        })

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="F1",
        ascending=False
    )

    print(
        "\n------ MODEL COMPARISON ------"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    return results_df, models


# 
# Save Best Model
# 

def save_best_model(
    results_df,
    models
):

    best_model_name = results_df.iloc[0]["Model"]

    best_model = models[
        best_model_name
    ]

    print(
        f"\nBest Model: {best_model_name}"
    )

    joblib.dump(
        best_model,
        "models/best_model.pkl"
    )

    print(
        "Best Model Saved Successfully!"
    )




if __name__ == "__main__":

    df = load_dataset()

    X, y = split_features_target(
        df
    )

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    results_df, models = compare_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    save_best_model(
        results_df,
        models
    )