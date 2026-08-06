import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(accuracy_score,precision_score,recall_score,f1_score)

#
# Load Dataset
# 
def load_dataset():
    print("Loading the ML dataset...")

    df=pd.read_csv("data/processed/ml_dataset.csv")

    return df

#
#Split features and target
#

'''
Machine learning models require:
    X : Input features (everything the model learns from).
    y : Target label (what the model should predict).
Here, fraud is the label, so it is separated from the rest of the dataset.
'''

def split_features_target(df):
    print("Splitting featues amd Target...")

    X = df.drop(columns=["fraud","risk_score","risk_severity"])      #removing fraud, risk_score and risk_severity columns from training
    y = df["fraud"]                     #keeping only the fraud column for testing

    return X,y

#
# Train and Test Slpit
#
'''
The dataset is divided into:

    80% Training Data : Used to teach the model.
    20% Testing Data : Used to evaluate how well the model performs on unseen invoices.

Using stratify=y ensures that the proportion of fraudulent and genuine invoices remains the same in both training and testing sets.
'''

def split_dataset(X,y):
    print("Creating Train/Test Split...")

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    return X_train, X_test, y_train, y_test


#
# Random Forest (Training)
#

'''
This function creates and trains a Random Forest Classifier using the training dataset.

A Random Forest builds many decision trees and combines their predictions. This usually gives better accuracy and reduces overfitting compared to a single Decision Tree.
'''

def train_random_forest(X_train, y_train):
    print("Training Random forest Model...")

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    return model

#
# Evaluate Model
#

def evaluate_model(model,X_test,y_test):
    print("Evaluating Model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test,predictions)
    precision=precision_score(y_test, predictions)
    recall = recall_score(y_test,predictions)
    f1=f1_score(y_test,predictions)


    print("\n----- MODEL PERFORMANCE -----")

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    return model


#
# Save Model
#
'''
After training, we save the model so we don't have to retrain it every time we want to make a prediction.

The saved model (fraud_model.pkl) will later be loaded by predict.py to classify new invoices.
'''
def save_model(model):

    print("\nSaving Model...")

    joblib.dump(model,"models/fraud_model.pkl")

    print("Model Saved Successfully!")

#Test
if __name__ == "__main__":

    df = load_dataset()

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = train_random_forest(X_train,y_train)

    evaluate_model(model,X_test,y_test)

    save_model(model)