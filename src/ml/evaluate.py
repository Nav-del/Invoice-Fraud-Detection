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
# Feature importance
#
'''
Random Forest assigns an importance score to every feature based on how much it contributes to making predictions. By ranking these scores, we can identify whether the model is relying on genuinely useful patterns or on features that directly reveal the fraud label.
'''
def feature_importance(model,df):
    print("\nTop important features\n")

    X = df.drop(columns=["fraud"])

    importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})

    importance = importance.sort_values(by = "Importance", ascending = False)

    #Graph for README
    plt.figure(figsize=(10,6))

    plt.barh(
        importance["Feature"][:15],
        importance["Importance"][:15]
    )

    plt.xlabel("Importance Score")

    plt.ylabel("Feature")

    plt.title("Top 15 Feature Importance")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.show()

    print(importance.head(15))

    return importance


# Test 
if __name__ == "__main__":

    model, df = load_resources()

    feature_importance(model, df)