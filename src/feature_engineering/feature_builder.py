"""
Feature Builder

Creates machine learning features from the
preprocessed invoice dataset.
"""

import os
import joblib
import pandas as pd
import numpy as np

# ----------------------------------------
# File Paths
# ----------------------------------------

INPUT_FILE = "data/processed/preprocessed_dataset.csv"

OUTPUT_FILE = "data/processed/ml_dataset.csv"

FEATURE_FILE = "models/feature_columns.pkl"

os.makedirs("models", exist_ok=True)

# ----------------------------------------
# Load Dataset
# ----------------------------------------

def load_dataset():

    print("\nLoading preprocessed dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows : {len(df)}")

    print(f"Columns : {len(df.columns)}")

    return df


# ----------------------------------------
# Invoice Age
# ----------------------------------------

def create_invoice_age(df):

    print("\nCreating Invoice Age...")

    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    today = pd.Timestamp.today()

    df["invoice_age"] = (

        today - df["invoice_date"]

    ).dt.days

    return df


# ----------------------------------------
# Payment Window
# ----------------------------------------

def create_payment_window(df):

    print("Creating Payment Window...")

    df["due_date"] = pd.to_datetime(df["due_date"])

    df["payment_window"] = (

        df["due_date"] - df["invoice_date"]

    ).dt.days

    return df


# ----------------------------------------
# Tax Ratio
# ----------------------------------------

def create_tax_ratio(df):

    print("Creating Tax Ratio...")

    df["tax_ratio"] = (

        df["tax_amount"]

        /

        df["invoice_amount"]

    ).round(4)

    return df


# ----------------------------------------
# Discount Ratio
# ----------------------------------------

def create_discount_ratio(df):

    print("Creating Discount Ratio...")

    df["discount_ratio"] = (

        df["discount_amount"]

        /

        df["invoice_amount"]

    ).round(4)

    return df


# ----------------------------------------
# Effective GST
# ----------------------------------------

def create_effective_gst(df):

    print("Creating Effective GST...")

    df["effective_gst"] = (

        df["tax_amount"]

        /

        df["invoice_amount"]

        *

        100

    ).round(2)

    return df

# ----------------------------------------
# Vendor Amount Ratio
# ----------------------------------------

def create_vendor_amount_ratio(df):

    print("Creating Vendor Amount Ratio...")

    vendor_avg = df.groupby("vendor_name")["invoice_amount"].transform("mean")

    df["vendor_amount_ratio"] = (
        df["invoice_amount"] / vendor_avg
    ).round(3)

    return df

if __name__ == "__main__":

    df = load_dataset()

    df = create_invoice_age(df)

    df = create_payment_window(df)

    df = create_tax_ratio(df)

    df = create_discount_ratio(df)

    df = create_effective_gst(df)

    df = create_vendor_amount_ratio(df) 

    print(df.head())