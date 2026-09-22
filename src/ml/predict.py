'''
Predict.py

Loads the trained model and predicts weather an invoice is fraud or not
'''

import joblib
import pandas as pd

#
# Load Model
#
def load_model():
    print("Loading Best Model...")

    model = joblib.load("models/best_model.pkl")

    return model

#
# Prepare Invoice
#
'''
This converts one invoice's information into the same DataFrame format expected by the model. We also remove the target and leakage features because those are not supposed to be provided to the model during prediction.
'''
def prepare_invoice(invoice_data, feature_columns):
    print("Preparing Invoice...")

    df = pd.DataFrame([invoice_data])

    df = df.drop(columns=["fraud","risk_score","risk_severity"],errors = "ignore")

    for column in feature_columns:      #Adding the missing features
        if column not in df.columns:
            df[column]=0
        df = df[feature_columns]        #Keep only the training features
    return df

#
# Predict fraud
#
'''
The model produces both a classification (Fraud or Genuine) and a fraud probability. The probability is important because later we'll give this value to Llama 3.2 so it can generate a meaningful explanation.
'''
def predict_invoice(model, invoice_data, feature_columns):
    df = prepare_invoice(invoice_data, feature_columns)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if prediction==1:
        result = "Fraud"
    else:
        result = "Genuine"

    print("------Prediction------")

    print(f"Result : {result}")
    print(f"Fraud Probability : {probability:.2%}")
    return result, probability

#
# Load Feature Columns
#
'''
During training, the model learned from a specific set and order of features. This function loads that saved feature list so a new invoice is arranged exactly the same way before prediction.
'''
def load_feature_columns():
    print("Loading the feature columns...")

    feature_columns = joblib.load("models/features_columns.pkl")

    return feature_columns

#
# Test 
#
if __name__ =="__main__":
    model = load_model()
    feature_columns = load_feature_columns()

    df = pd.read_csv("data/processed/ml_dataset.csv")

    sample_invoice = df.iloc[0].to_dict()

    predict_invoice(model,sample_invoice,feature_columns)