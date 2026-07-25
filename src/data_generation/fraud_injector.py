"""
Fraud Injector

Reads genuine invoices and injects realistic
fraud patterns to create a training dataset.
"""

import os
import random
import pandas as pd

from src.config import FRAUD_RATE, RISK_WEIGHTS

# -----------------------------------------
# File Paths
# -----------------------------------------

INPUT_FILE = "data/raw/genuine_invoices.csv"

OUTPUT_FILE = "data/processed/fraud_dataset.csv"

os.makedirs("data/processed", exist_ok=True)

# -----------------------------------------
# Load Dataset
# -----------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"\nLoaded {len(df)} genuine invoices.")

# -----------------------------------------
# Fraud Types
# -----------------------------------------

FRAUD_TYPES = [

    "duplicate_invoice",

    "duplicate_po",

    "invalid_gst",

    "tax_mismatch",

    "date_anomaly",

    "amount_anomaly",

    "payment_anomaly",

    "vendor_mismatch",

    "currency_mismatch"

]

# -----------------------------------------
# Risk Score Calculator
# -----------------------------------------

def calculate_risk(active_frauds):

    score = 0

    for fraud in active_frauds:

        score += RISK_WEIGHTS.get(fraud, 0)

    return min(score, 100)

# -----------------------------------------
# Fraud Label
# -----------------------------------------

def assign_label(score):

    if score >= 70:

        return 1

    return 0

# -----------------------------------------
# Initialize New Columns
# -----------------------------------------

new_columns = [

    "duplicate_invoice",

    "duplicate_po",

    "gst_valid",

    "tax_mismatch",

    "date_anomaly",

    "amount_anomaly",

    "payment_anomaly",

    "vendor_mismatch",

    "currency_mismatch",

    "risk_score",

    "fraud"

]

for col in new_columns:

    if col == "gst_valid":

        df[col] = 1

    else:

        df[col] = 0

print("Fraud columns initialized.")
# -----------------------------------------
# Inject Fraud
# -----------------------------------------

def inject_fraud(df):

    invoice_numbers = df["invoice_number"].tolist()

    purchase_orders = df["purchase_order"].tolist()

    total_rows = len(df)

    fraud_count = int(total_rows * FRAUD_RATE)

    fraud_indices = random.sample(
        range(total_rows),
        fraud_count
    )

    print(f"\nInjecting fraud into {fraud_count} invoices...\n")

    for idx in fraud_indices:

        active_frauds = []

        # Randomly choose 1-4 fraud indicators
        frauds = random.sample(
            FRAUD_TYPES,
            random.randint(1, 4)
        )

        # ---------------------------------
        # Duplicate Invoice
        # ---------------------------------

        if "duplicate_invoice" in frauds:

            duplicate = random.choice(invoice_numbers)

            df.at[idx, "invoice_number"] = duplicate

            df.at[idx, "duplicate_invoice"] = 1

            active_frauds.append("duplicate_invoice")

        # ---------------------------------
        # Duplicate Purchase Order
        # ---------------------------------

        if "duplicate_po" in frauds:

            duplicate = random.choice(purchase_orders)

            df.at[idx, "purchase_order"] = duplicate

            df.at[idx, "duplicate_po"] = 1

            active_frauds.append("duplicate_po")

        # ---------------------------------
        # Invalid GST
        # ---------------------------------

        if "invalid_gst" in frauds:

            df.at[idx, "gst_number"] = "INVALIDGST"

            df.at[idx, "gst_valid"] = 0

            active_frauds.append("invalid_gst")

        # ---------------------------------
        # Tax Mismatch
        # ---------------------------------

        if "tax_mismatch" in frauds:

            tax = df.at[idx, "tax_amount"]

            tax = round(
                tax * random.uniform(0.20, 2.50),
                2
            )

            df.at[idx, "tax_amount"] = tax

            df.at[idx, "tax_mismatch"] = 1

            active_frauds.append("tax_mismatch")

        # ---------------------------------
        # Date Anomaly
        # ---------------------------------

        if "date_anomaly" in frauds:

            invoice_date = pd.to_datetime(
                df.at[idx, "invoice_date"]
            )

            due_date = invoice_date - pd.Timedelta(
                days=random.randint(1, 20)
            )

            df.at[idx, "due_date"] = due_date.date()

            df.at[idx, "date_anomaly"] = 1

            active_frauds.append("date_anomaly")

        # ---------------------------------
        # Amount Spike
        # ---------------------------------

        if "amount_anomaly" in frauds:

            amount = df.at[idx, "invoice_amount"]

            amount *= random.uniform(2.0, 5.0)

            df.at[idx, "invoice_amount"] = round(
                amount,
                2
            )

            df.at[idx, "amount_anomaly"] = 1

            active_frauds.append("amount_anomaly")

        # ---------------------------------
        # Payment Anomaly
        # ---------------------------------

        if "payment_anomaly" in frauds:

            df.at[idx, "payment_status"] = "Failed"

            df.at[idx, "payment_anomaly"] = 1

            active_frauds.append("payment_anomaly")

        # ---------------------------------
        # Vendor Mismatch
        # ---------------------------------

        if "vendor_mismatch" in frauds:

            df.at[idx, "vendor_name"] = "Unknown Vendor"

            df.at[idx, "vendor_mismatch"] = 1

            active_frauds.append("vendor_mismatch")

        # ---------------------------------
        # Currency Mismatch
        # ---------------------------------

        if "currency_mismatch" in frauds:

            df.at[idx, "currency"] = random.choice(
                [
                    "USD",
                    "EUR",
                    "AED",
                    "GBP"
                ]
            )

            df.at[idx, "currency_mismatch"] = 1

            active_frauds.append("currency_mismatch")

        # ---------------------------------
        # Blacklisted Vendor
        # ---------------------------------

        if df.at[idx, "blacklisted"]:

            active_frauds.append(
                "blacklisted_vendor"
            )

        # ---------------------------------
        # Risk Score
        # ---------------------------------

        score = calculate_risk(
            active_frauds
        )

        df.at[idx, "risk_score"] = score

        df.at[idx, "fraud"] = assign_label(
            score
        )

    print("Fraud Injection Complete.\n")

    return df
# -----------------------------------------
# Inject Fraud
# -----------------------------------------

def inject_fraud(df):

    invoice_numbers = df["invoice_number"].tolist()

    purchase_orders = df["purchase_order"].tolist()

    total_rows = len(df)

    fraud_count = int(total_rows * FRAUD_RATE)

    fraud_indices = random.sample(
        range(total_rows),
        fraud_count
    )

    print(f"\nInjecting fraud into {fraud_count} invoices...\n")

    for idx in fraud_indices:

        active_frauds = []

        # Randomly choose 1-4 fraud indicators
        frauds = random.sample(
            FRAUD_TYPES,
            random.randint(1, 4)
        )

        # ---------------------------------
        # Duplicate Invoice
        # ---------------------------------

        if "duplicate_invoice" in frauds:

            duplicate = random.choice(invoice_numbers)

            df.at[idx, "invoice_number"] = duplicate

            df.at[idx, "duplicate_invoice"] = 1

            active_frauds.append("duplicate_invoice")

        # ---------------------------------
        # Duplicate Purchase Order
        # ---------------------------------

        if "duplicate_po" in frauds:

            duplicate = random.choice(purchase_orders)

            df.at[idx, "purchase_order"] = duplicate

            df.at[idx, "duplicate_po"] = 1

            active_frauds.append("duplicate_po")

        # ---------------------------------
        # Invalid GST
        # ---------------------------------

        if "invalid_gst" in frauds:

            df.at[idx, "gst_number"] = "INVALIDGST"

            df.at[idx, "gst_valid"] = 0

            active_frauds.append("invalid_gst")

        # ---------------------------------
        # Tax Mismatch
        # ---------------------------------

        if "tax_mismatch" in frauds:

            tax = df.at[idx, "tax_amount"]

            tax = round(
                tax * random.uniform(0.20, 2.50),
                2
            )

            df.at[idx, "tax_amount"] = tax

            df.at[idx, "tax_mismatch"] = 1

            active_frauds.append("tax_mismatch")

        # ---------------------------------
        # Date Anomaly
        # ---------------------------------

        if "date_anomaly" in frauds:

            invoice_date = pd.to_datetime(
                df.at[idx, "invoice_date"]
            )

            due_date = invoice_date - pd.Timedelta(
                days=random.randint(1, 20)
            )

            df.at[idx, "due_date"] = due_date.date()

            df.at[idx, "date_anomaly"] = 1

            active_frauds.append("date_anomaly")

        # ---------------------------------
        # Amount Spike
        # ---------------------------------

        if "amount_anomaly" in frauds:

            amount = df.at[idx, "invoice_amount"]

            amount *= random.uniform(2.0, 5.0)

            df.at[idx, "invoice_amount"] = round(
                amount,
                2
            )

            df.at[idx, "amount_anomaly"] = 1

            active_frauds.append("amount_anomaly")

        # ---------------------------------
        # Payment Anomaly
        # ---------------------------------

        if "payment_anomaly" in frauds:

            df.at[idx, "payment_status"] = "Failed"

            df.at[idx, "payment_anomaly"] = 1

            active_frauds.append("payment_anomaly")

        # ---------------------------------
        # Vendor Mismatch
        # ---------------------------------

        if "vendor_mismatch" in frauds:

            df.at[idx, "vendor_name"] = "Unknown Vendor"

            df.at[idx, "vendor_mismatch"] = 1

            active_frauds.append("vendor_mismatch")

        # ---------------------------------
        # Currency Mismatch
        # ---------------------------------

        if "currency_mismatch" in frauds:

            df.at[idx, "currency"] = random.choice(
                [
                    "USD",
                    "EUR",
                    "AED",
                    "GBP"
                ]
            )

            df.at[idx, "currency_mismatch"] = 1

            active_frauds.append("currency_mismatch")

        # ---------------------------------
        # Blacklisted Vendor
        # ---------------------------------

        if df.at[idx, "blacklisted"]:

            active_frauds.append(
                "blacklisted_vendor"
            )

        # ---------------------------------
        # Risk Score
        # ---------------------------------

        score = calculate_risk(
            active_frauds
        )

        df.at[idx, "risk_score"] = score

        df.at[idx, "fraud"] = assign_label(
            score
        )

    print("Fraud Injection Complete.\n")

    return df
# -----------------------------------------
# Fraud Statistics
# -----------------------------------------

def print_statistics(df):

    print("\n===================================")
    print("FRAUD DATASET SUMMARY")
    print("===================================\n")

    print(f"Total Records           : {len(df)}")

    fraud_count = df["fraud"].sum()

    genuine_count = len(df) - fraud_count

    print(f"Genuine Invoices        : {genuine_count}")

    print(f"Fraudulent Invoices     : {fraud_count}")

    print("\nFraud Distribution")

    print(df["fraud"].value_counts())

    print("\nAverage Risk Score")

    print(round(df["risk_score"].mean(), 2))

    print("\nFraud Indicators")

    indicators = [

        "duplicate_invoice",
        "duplicate_po",
        "gst_valid",
        "tax_mismatch",
        "date_anomaly",
        "amount_anomaly",
        "payment_anomaly",
        "vendor_mismatch",
        "currency_mismatch"

    ]

    for col in indicators:

        if col == "gst_valid":

            count = (df[col] == 0).sum()

        else:

            count = df[col].sum()

        print(f"{col:20}: {count}")

    print("\n===================================\n")


# -----------------------------------------
# Save Dataset
# -----------------------------------------

def save_dataset(df):

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nDataset Saved Successfully!")

    print(f"Location : {OUTPUT_FILE}")


# -----------------------------------------
# Main
# -----------------------------------------

if __name__ == "__main__":

    df = inject_fraud(df)

    print_statistics(df)

    save_dataset(df)

    print("\nSample Records\n")

    print(df.head())

    print("\nFraud Injection Completed Successfully!")