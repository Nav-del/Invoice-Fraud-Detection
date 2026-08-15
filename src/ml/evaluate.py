import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import(confusion_matrix,classification_report,ConfusionMatrixDisplay)

#
# Load model and dataset
#
'''
We load the trained model and the processed dataset so we can analyze how the model behaves and inspect feature importance.
'''
def load_resources():
    print("Loading Model...")

    model =joblib.load("models/fraud_model.pkl")

    df = pd.read_csv("data/processed/ml_dataset.csv")

    return model, df


#
# Prepare test data
#
'''
We recreate the same 80/20 split used during training so that evaluation happens on the exact type of unseen data the model was tested on.
We also remove risk_score and risk_severity here, keeping evaluation consistent with the leakage fix in train.py.
'''
def prepare_test_data(df):
    print("Preparing test data...")

    X = df.drop(columns=["fraud","risk_score","risk_severity"])

    y = df["fraud"]

    from sklearn.model_selection import train_test_split

    _, X_test,_,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    return X_test, y_test


#
# Feature importance
#
'''
Random Forest assigns an importance score to every feature based on how much it contributes to making predictions. By ranking these scores, we can identify whether the model is relying on genuinely useful patterns or on features that directly reveal the fraud label.

The feature list used for evaluation must be identical to the feature list used during training. Since we removed risk_score and risk_severity to prevent leakage, they must also be excluded when calculating feature importance.
'''
def feature_importance(model, df):

    print("\nTop important features\n")

    X = df.drop(
        columns=[
            "fraud",
            "risk_score",
            "risk_severity"
        ]
    )

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(importance.head(15))

    return importance


#
# Confusion Matrix
#
'''
The confusion matrix shows exactly how many invoices were correctly and incorrectly classified as Genuine or Fraud.
This is especially important for fraud detection because we want to see how many fraudulent invoices the model is actually missing.
'''
def create_confusion_matrix(model, X_test, y_test):

    print("\nCreating Confusion Matrix...")

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nConfusion Matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Genuine", "Fraud"]
    )

    display.plot()

    plt.title("Invoice Fraud Detection - Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        "reports/confusion_matrix.png"
    )

    plt.close()

    print("Confusion Matrix saved.")

#
# Classification Report
#
'''
This gives us precision, recall, F1-score, and the number of samples for both Genuine and Fraud classes. Saving it as a text file gives us a permanent evaluation artifact that we can later reference in the README.
'''
def create_classification_report(model, X_test, y_test):

    print("\nCreating Classification Report...")

    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Genuine", "Fraud"]
    )

    print("\nClassification Report:")
    print(report)

    with open(
        "reports/classification_report.txt",
        "w"
    ) as file:

        file.write(report)

    print("Classification Report saved.")


if __name__ == "__main__":

    model, df = load_resources()

    X_test, y_test = prepare_test_data(df)

    feature_importance(
        model,
        df
    )

    create_confusion_matrix(
        model,
        X_test,
        y_test
    )

    create_classification_report(
        model,
        X_test,
        y_test
    )