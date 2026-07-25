"""
Financial Feature Engineering

Creates financial features for invoice fraud detection.
"""

import numpy as np
import pandas as pd


# -----------------------------------------
# Tax Ratio
# -----------------------------------------

def create_tax_ratio(df):

    print("Creating Tax Ratio...")

    df["tax_ratio"] = (
        df["tax_amount"] /
        df["invoice_amount"]
    ).round(4)

    return df


# -----------------------------------------
# Discount Ratio
# -----------------------------------------

def create_discount_ratio(df):

    print("Creating Discount Ratio...")

    df["discount_ratio"] = (
        df["discount_amount"] /
        df["invoice_amount"]
    ).round(4)

    return df


# -----------------------------------------
# Effective GST
# -----------------------------------------

def create_effective_gst(df):

    print("Creating Effective GST...")

    df["effective_gst"] = (
        df["tax_amount"] /
        df["invoice_amount"]
        * 100
    ).round(2)

    return df

# -----------------------------------------
# Vendor Amount Ratio
# -----------------------------------------

def create_vendor_amount_ratio(df):

    print("Creating Vendor Amount Ratio...")

    vendor_average = (

        df.groupby("vendor_name")["invoice_amount"]

        .transform("mean")

    )

    df["vendor_amount_ratio"] = (

        df["invoice_amount"] /

        vendor_average

    ).round(3)

    return df


# -----------------------------------------
# High Value Invoice
# -----------------------------------------

def create_high_value_invoice(df):

    print("Creating High Value Invoice...")

    category_average = (

        df.groupby("category")["invoice_amount"]

        .transform("mean")

    )

    df["high_value_invoice"] = (

        df["invoice_amount"] >

        category_average

    ).astype(int)

    return df

# -----------------------------------------
# Amount Z-Score
# -----------------------------------------

def create_amount_zscore(df):

    print("Creating Amount Z-Score...")

    vendor_mean = (

        df.groupby("vendor_name")["invoice_amount"]

        .transform("mean")

    )

    vendor_std = (

        df.groupby("vendor_name")["invoice_amount"]

        .transform("std")

        .fillna(1)

    )

    vendor_std = vendor_std.replace(0, 1)

    df["amount_zscore"] = (

        (df["invoice_amount"] - vendor_mean)

        /

        vendor_std

    ).round(3)

    return df

#
# Create all finiacial features
#

def create_financial_features(df):

    print("\nCreating Financial Features...\n")

    df = create_tax_ratio(df)

    df = create_discount_ratio(df)

    df = create_effective_gst(df)

    df = create_vendor_amount_ratio(df)

    df = create_high_value_invoice(df)

    df = create_amount_zscore(df)

    return df


if __name__ == "__main__":

    df = pd.read_csv("data/processed/fraud_dataset.csv")

    df = create_financial_features(df)

    print("\nFinancial Features Created Successfully!\n")

    print(
        df[
            [
                "tax_ratio",
                "discount_ratio",
                "effective_gst",
                "vendor_amount_ratio",
                "high_value_invoice",
                "amount_zscore",
            ]
        ].head()
    )

    print("\nDataset Shape:", df.shape)

