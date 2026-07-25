'''
Temporal Feature Engineering
This file creates time-based features for the invoice fraud detection
'''

import pandas as pd

#
# Convert Dates
#

def convert_dates(df):
    print("Converting date columns : ")

    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    df["due_date"] = pd.to_datetime(df["due_date"])

    return df

#
# Invoice Age
#

def create_invoice_age(df):

    print("Creating Invoice Age : ")

    df["invoice_age"] = (
        df["due_date"] - df["invoice_date"]
    ).dt.days

    return df

#
# Invoice Month
#

def create_invoice_month(df):

    print("Creating Invoice Month : ")

    df["invoice_month"] = (
        df["invoice_date"]
        .dt.month
    )

    return df

#
# Invoice Quarter
#

def create_invoice_quarter(df):

    print("Creating Invoice Quarter : ")

    df["invoice_quarter"] = (
        df["invoice_date"]
        .dt.quarter
    )

    return df

#
# Invoice Day
#

def create_invoice_day(df):

    print("Creating Invoice Day : ")

    df["invoice_day"] = (
        df["invoice_date"]
        .dt.day
    )

    return df

#
# Weekend Invoice
#

def create_weekend_invoice(df):

    print("Creating Weekend Feature : ")

    df["is_weekend"] = (
        df["invoice_date"]
        .dt.weekday >=5
    ).astype(int)

    return df

#
# Month End
#

def create_month_end_feature(df):

    print("Creating Month End Feature : ")

    df["is_month_end"] = (
        df["invoice_date"]
        .dt.is_month_end
    ).astype(int)

    return df

#
# Wrapper
#

def create_temporal_features(df):

    print("\nCreating Temporal Features...\n")

    df = convert_dates(df)

    df = create_invoice_age(df)

    df = create_invoice_month(df)

    df = create_invoice_quarter(df)

    df = create_invoice_day(df)

    df = create_weekend_invoice(df)

    df = create_month_end_feature(df)

    return df

#
# Test
#

if __name__ == "__main__":

    df = pd.read_csv("data/processed/fraud_dataset.csv")

    df = create_temporal_features(df)

    print("\nTemporal Features Created Successfully!\n")

    print(

        df[

            [

                "invoice_age",

                "invoice_month",

                "invoice_quarter",

                "invoice_day",

                "is_weekend",

                "is_month_end",

            ]

        ].head()

    )

    print("\nDataset Shape:", df.shape)