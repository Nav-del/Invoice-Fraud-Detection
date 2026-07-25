"""
Invoice Generator
-----------------
Generates genuine invoices using the vendor master.
No fraud is injected in this module.
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import (
    TOTAL_INVOICES,
    GST_RATES,
    PAYMENT_STATUS,
    PAYMENT_TERMS,
    CURRENCY,
    DESCRIPTIONS
)

# -----------------------------------------
# File Paths
# -----------------------------------------

VENDOR_FILE = "data/raw/vendors.csv"
OUTPUT_FILE = "data/raw/genuine_invoices.csv"

os.makedirs("data/raw", exist_ok=True)

# -----------------------------------------
# Load Vendor Master
# -----------------------------------------

vendors = pd.read_csv(VENDOR_FILE)

print(f"Loaded {len(vendors)} vendors.")

# -----------------------------------------
# Weighted Vendor Selection
# -----------------------------------------

def select_vendor():
    """
    Select vendors based on invoice_frequency.
    Vendors with higher frequency are more likely
    to appear.
    """
    weights = vendors["invoice_frequency"].values
    return vendors.sample(
        weights=weights,
        n=1
    ).iloc[0]

# -----------------------------------------
# Invoice Number Generator
# -----------------------------------------

invoice_counter = 1

def generate_invoice_number():
    global invoice_counter

    invoice = f"INV-{datetime.now().year}-{invoice_counter:06d}"

    invoice_counter += 1

    return invoice

# -----------------------------------------
# Purchase Order Generator
# -----------------------------------------

po_counter = 1

def generate_purchase_order():
    global po_counter

    po = f"PO-{datetime.now().year}-{po_counter:06d}"

    po_counter += 1

    return po

# -----------------------------------------
# Amount Generator
# -----------------------------------------

def generate_invoice_amount(avg_amount):
    """
    Generate invoice amount around the vendor's
    average invoice value using a Gaussian
    distribution.
    """

    amount = np.random.normal(
        loc=avg_amount,
        scale=avg_amount * 0.15
    )

    return round(max(amount, 1000), 2)

# -----------------------------------------
# Discount Generator
# -----------------------------------------

def generate_discount():

    return random.choice([0, 5, 10])

# -----------------------------------------
# Payment Status
# -----------------------------------------

def generate_payment_status():

    return random.choice(PAYMENT_STATUS)

# -----------------------------------------
# Payment Terms
# -----------------------------------------

def generate_payment_terms():

    return random.choice(PAYMENT_TERMS)

# -----------------------------------------
# Description Generator
# -----------------------------------------

def generate_description():

    return random.choice(DESCRIPTIONS)

# -----------------------------------------
# GST Calculation
# -----------------------------------------

def calculate_tax(invoice_amount, category):
    """
    Calculate GST based on category.
    """

    gst_percentage = GST_RATES.get(category, 18)

    tax_amount = round(
        invoice_amount * (gst_percentage / 100),
        2
    )

    return gst_percentage, tax_amount


# -----------------------------------------
# Discount Calculation
# -----------------------------------------

def calculate_discount(invoice_amount, discount_percent):
    """
    Calculate discount amount.
    """

    return round(
        invoice_amount * (discount_percent / 100),
        2
    )


# -----------------------------------------
# Total Calculation
# -----------------------------------------

def calculate_totals(
    invoice_amount,
    tax_amount,
    discount_amount
):
    """
    Calculate subtotal and total.
    """

    subtotal = round(
        invoice_amount - discount_amount,
        2
    )

    total = round(
        subtotal + tax_amount,
        2
    )

    return subtotal, total


# -----------------------------------------
# Invoice Date
# -----------------------------------------

def generate_invoice_date():
    """
    Random invoice date within last year.
    """

    return (
        datetime.today()
        - timedelta(
            days=random.randint(0, 365)
        )
    ).date()


# -----------------------------------------
# Due Date
# -----------------------------------------

def generate_due_date(
    invoice_date,
    payment_terms
):
    """
    Calculate due date.
    """

    return (
        invoice_date
        + timedelta(days=payment_terms)
    )


# -----------------------------------------
# Currency
# -----------------------------------------

def generate_currency():

    return CURRENCY


# -----------------------------------------
# Payment Method
# -----------------------------------------

def generate_payment_method(vendor):

    return vendor["preferred_payment"]


# -----------------------------------------
# Invoice Object Generator
# -----------------------------------------

def create_invoice(vendor):
    """
    Generate one genuine invoice.
    """

    invoice_number = generate_invoice_number()

    purchase_order = generate_purchase_order()

    invoice_amount = generate_invoice_amount(
        vendor["average_invoice_amount"]
    )

    gst_percentage, tax_amount = calculate_tax(
        invoice_amount,
        vendor["category"]
    )

    discount_percent = generate_discount()

    discount_amount = calculate_discount(
        invoice_amount,
        discount_percent
    )

    subtotal, total_amount = calculate_totals(
        invoice_amount,
        tax_amount,
        discount_amount
    )

    payment_terms = generate_payment_terms()

    invoice_date = generate_invoice_date()

    due_date = generate_due_date(
        invoice_date,
        payment_terms
    )

    invoice = {

        "invoice_number": invoice_number,

        "purchase_order": purchase_order,

        "vendor_id": vendor["vendor_id"],

        "vendor_name": vendor["vendor_name"],

        "gst_number": vendor["gst_number"],

        "category": vendor["category"],

        "country": vendor["country"],

        "state": vendor["state"],

        "currency": generate_currency(),

        "invoice_amount": invoice_amount,

        "gst_percentage": gst_percentage,

        "tax_amount": tax_amount,

        "discount_percent": discount_percent,

        "discount_amount": discount_amount,

        "subtotal": subtotal,

        "total_amount": total_amount,

        "payment_method": generate_payment_method(vendor),

        "payment_terms": payment_terms,

        "payment_status": generate_payment_status(),

        "invoice_date": invoice_date,

        "due_date": due_date,

        "invoice_description": generate_description(),

        "vendor_frequency": vendor["invoice_frequency"],

        "vendor_risk": vendor["risk_level"],

        "blacklisted": vendor["blacklisted"]

    }

    return invoice

# -----------------------------------------
# Generate All Genuine Invoices
# -----------------------------------------

def generate_invoices():

    invoices = []

    print("\nGenerating Genuine Invoices...\n")

    for i in range(TOTAL_INVOICES):

        vendor = select_vendor()

        invoice = create_invoice(vendor)

        invoices.append(invoice)

        if (i + 1) % 1000 == 0:

            print(
                f"{i+1} invoices generated..."
            )

    return pd.DataFrame(invoices)


# -----------------------------------------
# Dataset Validation
# -----------------------------------------

def validate_dataset(df):

    print("\n==============================")
    print("DATASET VALIDATION")
    print("==============================")

    duplicate_invoice = df[
        "invoice_number"
    ].duplicated().sum()

    duplicate_po = df[
        "purchase_order"
    ].duplicated().sum()

    print(
        f"Duplicate Invoice Numbers : {duplicate_invoice}"
    )

    print(
        f"Duplicate Purchase Orders : {duplicate_po}"
    )

    print(
        f"Total Records : {len(df)}"
    )

    print(
        f"Unique Vendors : {df['vendor_name'].nunique()}"
    )

    print(
        f"Average Invoice Amount : ₹{round(df['invoice_amount'].mean(),2)}"
    )

    print(
        f"Average Tax Amount : ₹{round(df['tax_amount'].mean(),2)}"
    )

    print(
        f"Blacklisted Vendor Invoices : {df['blacklisted'].sum()}"
    )

    print("==============================\n")


# -----------------------------------------
# Save Dataset
# -----------------------------------------

def save_dataset(df):

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nDataset saved successfully!")

    print(f"\nLocation : {OUTPUT_FILE}")


# -----------------------------------------
# Main
# -----------------------------------------

if __name__ == "__main__":

    invoice_df = generate_invoices()

    validate_dataset(invoice_df)

    save_dataset(invoice_df)

    print("\nSample Records\n")

    print(invoice_df.head())

    print("\nInvoice Generation Complete!")