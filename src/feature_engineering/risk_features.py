"""
Risk Feature Engineering

This creates a high level fraud risk features combining multiple fraud indicators
"""

import pandas as pd
import numpy as np

#
# Financial risk count
#
'''
Explanation
An invoice can have several financial problems such as an invalid GST number, incorrect tax calculation, or an unusually high/low invoice amount. Instead of checking these separately, we count how many financial issues are present in one invoice.

Why?
Instead of making the model learn three separate columns every time, we give it a summary of the invoice's financial health.
'''
def create_financial_risk_count(df):
    print("Creating Financial risk count : ")

    df["financial_risk_count"] = (
        (1 - df["gst_valid"]) + df["tax_mismatch"] + df["amount_anomaly"]
    )

    return df

#
# Vendor risk count
#
'''
Explanation
This function evaluates how trustworthy the vendor is. It checks whether the vendor information doesn't match, whether the vendor is blacklisted, and whether they are already classified as a high-risk vendor.

Why?
Fraud often originates from suspicious vendors. Combining these checks into one feature helps the model identify vendor-related fraud patterns more easily.
'''
def create_vendor_risk_count(df):
    print("Creating Vendor Risk Count : ")

    df["vendor_risk_count"] = (
        df["vendor_mismatch"] + df["blacklisted"] + (df["vendor_risk"]=="High").astype(int)
    )

    return df

#
# Opperational Risk Count
#
'''
Explanation
Operational fraud often happens because of duplicate invoices, repeated purchase orders, payment irregularities, incorrect dates, or currency inconsistencies. This function counts all these operational issues into a single feature.

Why?
Operational errors and duplicate records are common indicators of invoice fraud, making this a strong predictive feature.
'''
def create_operational_risk_count(df):
    print("Creating Operational risk count : ")

    df["operational_risk_count"] = (
        df["duplicate_invoice"] + df["duplicate_po"] + df["payment_anomaly"] + df["date_anomaly"] + df["currency_mismatch"]
    )

    return df

#
# Total risk indicators
#
'''
Explanation
So far we've measured risk in three separate categories: Financial, Vendor, and Operational. This function combines them into one overall score representing the total number of warning signs.

Why?
Instead of looking at financial, vendor, and operational risks separately, the model gets a quick summary of how many total warning signs an invoice has.
'''
def create_total_risk_indicators(df):

    print("Creating Total risk indicators : ")

    df["total_risk_indicators"] = (
        df["financial_risk_count"] + df["vendor_risk_count"] + df["operational_risk_count"]
    )

    return df


#
# High Risk vendor
#
'''
Explanation
This function converts the vendor's risk level into a simple binary flag. If the vendor is marked as High Risk, the value becomes 1; otherwise, it becomes 0.

Why?
Binary features are easier for tree-based models to split on, making vendor-related fraud patterns easier to learn.
'''
def create_high_risk_vendor(df):

    print("Creating High risk vendor flag : ")

    df["high_risk_vendor"] = (
        df["vendor_risk"] == "High"
    ).astype(int)

    return df

#
# Blacklisted Vendor
#
'''
Explanation
This function explicitly identifies whether the vendor belongs to a blacklist. Although this information already exists, creating a dedicated binary feature makes it easier to use in later analysis and reporting.

Why?
Blacklisted vendors are inherently suspicious, so making this an explicit feature helps the model recognize high-risk vendors quickly.
'''
def create_blacklisted_vendor(df):

    print("Creating Blacklisted Vendor Flag...")

    df["blacklisted_vendor"] = (

        df["blacklisted"]

    ).astype(int)

    return df

#
# Risk Severity
#
'''
Explanation
The original risk_score is a numerical value that may not be easy to interpret. This function converts that score into human-readable categories: Low, Medium, or High.

Why?
Categorical severity levels are easier to interpret and later allow the Llama explainability module to generate more natural fraud reports.
'''
def create_risk_severity(df):

    print("Creating Risk Severity...")

    conditions = [

        df["risk_score"] <= 30,

        (df["risk_score"] > 30) & (df["risk_score"] <= 60),

        df["risk_score"] > 60

    ]

    labels = [

        "Low",

        "Medium",

        "High"

    ]

    df["risk_severity"] = np.select(

        conditions,

        labels,

        default="Low"

    )

    return df

#
# Wrapper Function
#

def create_risk_features(df):

    print("\nCreating Risk Features...\n")

    df = create_financial_risk_count(df)

    df = create_vendor_risk_count(df)

    df = create_operational_risk_count(df)

    df = create_total_risk_indicators(df)

    df = create_high_risk_vendor(df)

    df = create_blacklisted_vendor(df)

    df = create_risk_severity(df)

    return df

if __name__ == "__main__":

    df = pd.read_csv("data/processed/fraud_dataset.csv")

    df = create_risk_features(df)

    print("\nRisk Features Created Successfully!\n")

    print(

        df[

            [

                "financial_risk_count",

                "vendor_risk_count",

                "operational_risk_count",

                "total_risk_indicators",

                "high_risk_vendor",

                "blacklisted_vendor",

                "risk_severity"

            ]

        ].head()

    )

    print("\nDataset Shape:", df.shape)