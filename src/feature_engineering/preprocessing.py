print("Preprocessing.py started")
"""
Preprocessing Module

Responsible for:
1. Loading the fraud dataset
2. Cleaning missing values
3. Removing duplicate rows
4. Basic data validation

NOTE:
This module DOES NOT create ML features.
Feature engineering is handled separately.
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# ------------------------------------------
# File Paths
# ------------------------------------------

INPUT_FILE = "data/processed/fraud_dataset.csv"

OUTPUT_FILE = "data/processed/preprocessed_dataset.csv"

ENCODER_FILE = "models/ordinal_encoder.pkl"

os.makedirs("models", exist_ok=True)

os.makedirs("data/processed", exist_ok=True)

# ------------------------------------------
# Load Dataset
# ------------------------------------------

def load_dataset():

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset Loaded Successfully")

    print(f"Rows : {len(df)}")

    print(f"Columns : {len(df.columns)}")

    return df


# ------------------------------------------
# Remove Duplicate Rows
# ------------------------------------------

def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    removed = before - len(df)

    print(f"\nDuplicate Rows Removed : {removed}")

    return df


# ------------------------------------------
# Handle Missing Values
# ------------------------------------------

def handle_missing_values(df):

    print("\nHandling Missing Values...")

    numeric_columns = df.select_dtypes(
        include=[
            np.number
        ]
    ).columns

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns

    # -----------------------------
    # Numeric Columns
    # -----------------------------

    numeric_imputer = SimpleImputer(
        strategy="median"
    )

    df[numeric_columns] = numeric_imputer.fit_transform(
        df[numeric_columns]
    )

    # -----------------------------
    # Categorical Columns
    # -----------------------------

    categorical_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    df[categorical_columns] = categorical_imputer.fit_transform(
        df[categorical_columns]
    )

    print("Missing Values Filled Successfully")

    return df



# ------------------------------------------
# Dataset Summary
# ------------------------------------------

def dataset_summary(df):

    print("\n==============================")

    print("DATASET SUMMARY")

    print("==============================")

    print(f"Rows      : {len(df)}")

    print(f"Columns   : {len(df.columns)}")

    print(f"Null Values : {df.isnull().sum().sum()}")

    print("==============================\n")


# ------------------------------------------
# Convert Date Columns
# ------------------------------------------

def convert_dates(df):

    print("\nConverting Date Columns...")

    date_columns = [

        "invoice_date",

        "due_date"

    ]

    for col in date_columns:

        if col in df.columns:

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )

    print("Date Conversion Complete")

    return df


# ------------------------------------------
# Encode Categorical Columns
# ------------------------------------------

def encode_categorical_columns(df):

    print("\nEncoding Categorical Columns...")

    categorical_columns = [

        "vendor_name",

        "vendor_id",

        "category",

        "country",

        "state",

        "currency",

        "payment_method",

        "payment_status",

        "vendor_risk"

    ]

    # Only encode columns that actually exist
    categorical_columns = [
        col for col in categorical_columns
        if col in df.columns
    ]

    encoder = OrdinalEncoder(

        handle_unknown="use_encoded_value",

        unknown_value=-1

    )

    df[categorical_columns] = encoder.fit_transform(

        df[categorical_columns]

    )

    print(f"Encoded {len(categorical_columns)} categorical columns.")

    return df, encoder, categorical_columns


# ------------------------------------------
# Save Encoder
# ------------------------------------------

def save_encoder(encoder, categorical_columns):

    encoder_data = {

        "encoder": encoder,

        "columns": categorical_columns

    }

    joblib.dump(

        encoder_data,

        ENCODER_FILE

    )

    print("\nEncoder Saved Successfully")

    print(f"Location : {ENCODER_FILE}")


# ------------------------------------------
# Save Processed Dataset
# ------------------------------------------

def save_dataset(df):

    df.to_csv(

        OUTPUT_FILE,

        index=False

    )

    print("\nPreprocessed Dataset Saved")

    print(f"Location : {OUTPUT_FILE}")


# ------------------------------------------
# Dataset Validation
# ------------------------------------------

def validate_dataset(df):

    print("\n==============================")

    print("PREPROCESSING SUMMARY")

    print("==============================")

    print(f"Rows : {len(df)}")

    print(f"Columns : {len(df.columns)}")

    print(f"Null Values : {df.isnull().sum().sum()}")

    print(f"Duplicate Rows : {df.duplicated().sum()}")

    print("\nColumn Types")

    print(df.dtypes)

    print("\nSample Records")

    print(df.head())

    print("\n==============================")


# ------------------------------------------
# Main
# ------------------------------------------

if __name__ == "__main__":

    print("\n========== PREPROCESSING ==========\n")

    df = load_dataset()

    df = remove_duplicates(df)

    df = handle_missing_values(df)

    df = convert_dates(df)

    df, encoder, categorical_columns = encode_categorical_columns(df)

    save_encoder(

        encoder,

        categorical_columns

    )

    save_dataset(df)

    validate_dataset(df)

    print("\nPreprocessing Completed Successfully!\n")