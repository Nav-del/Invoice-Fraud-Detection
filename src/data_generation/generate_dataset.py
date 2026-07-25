import pandas as pd
import random
from datetime import datetime, timedelta

# ----------------------------------
# Configuration
# ----------------------------------
NUM_RECORDS = 10000
FRAUD_RATE = 0.30

vendors = [
    "ABC Technologies",
    "XYZ Solutions",
    "Global Traders",
    "Prime Electronics",
    "TechNova",
    "Skyline Pvt Ltd",
    "Vertex Systems",
    "Bluechip Enterprises",
    "Apex Supplies",
    "Future Innovations"
]

categories = [
    "Electronics",
    "Software",
    "Hardware",
    "Consulting",
    "Maintenance"
]

payment_methods = [
    "Bank Transfer",
    "UPI",
    "Cheque",
    "Credit Card"
]

data = []

# ----------------------------------
# GST Generator
# ----------------------------------
def generate_gst():
    return (
        str(random.randint(10, 37))
        + ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))
        + ''.join(random.choices('0123456789', k=4))
        + random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        + random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        + 'Z'
        + random.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    )

# ----------------------------------
# Dataset Generation
# ----------------------------------
for i in range(NUM_RECORDS):

    invoice_number = f"INV-{100000 + i}"

    vendor_name = random.choice(vendors)

    gst_number = generate_gst()

    invoice_category = random.choice(categories)

    payment_method = random.choice(payment_methods)

    invoice_amount = round(
        random.uniform(1000, 100000),
        2
    )

    tax_amount = round(
        invoice_amount * 0.18,
        2
    )

    payment_terms = random.choice(
        [15, 30, 45, 60]
    )

    invoice_date = (
        datetime.today()
        - timedelta(
            days=random.randint(0, 365)
        )
    )

    due_date = (
        invoice_date
        + timedelta(days=payment_terms)
    )

    vendor_frequency = random.randint(
        1,
        100
    )

    # -----------------------------
    # Default Genuine Values
    # -----------------------------
    duplicate_invoice = 0
    gst_valid = 1
    tax_mismatch = 0
    date_anomaly = 0

    amount_deviation = round(
        random.uniform(0.5, 1.5),
        2
    )

    fraud = 0

    # -----------------------------
    # Inject Fraud
    # -----------------------------
    if random.random() < FRAUD_RATE:

        fraud = 1

        fraud_type = random.choice([
            "duplicate",
            "gst",
            "tax",
            "amount",
            "date"
        ])

        if fraud_type == "duplicate":

            duplicate_invoice = 1

        elif fraud_type == "gst":

            gst_valid = 0

            gst_number = random.choice([
                "INVALIDGST",
                "GST123",
                "12345",
                "ABCDE",
                "GST-ERROR"
            ])

        elif fraud_type == "tax":

            tax_mismatch = 1

            tax_amount = round(
                invoice_amount *
                random.uniform(0.01, 0.50),
                2
            )

        elif fraud_type == "amount":

            amount_deviation = round(
                random.uniform(2.0, 8.0),
                2
            )

        elif fraud_type == "date":

            date_anomaly = 1

            due_date = (
                invoice_date
                - timedelta(
                    days=random.randint(1, 30)
                )
            )

    # -----------------------------
    # Add Record
    # -----------------------------
    data.append([
        invoice_number,
        vendor_name,
        gst_number,
        invoice_category,
        payment_method,
        invoice_amount,
        tax_amount,
        payment_terms,
        invoice_date,
        due_date,
        vendor_frequency,
        duplicate_invoice,
        gst_valid,
        tax_mismatch,
        date_anomaly,
        amount_deviation,
        fraud
    ])

# ----------------------------------
# Create DataFrame
# ----------------------------------
columns = [
    "invoice_number",
    "vendor_name",
    "gst_number",
    "invoice_category",
    "payment_method",
    "invoice_amount",
    "tax_amount",
    "payment_terms",
    "invoice_date",
    "due_date",
    "vendor_frequency",
    "duplicate_invoice",
    "gst_valid",
    "tax_mismatch",
    "date_anomaly",
    "amount_deviation",
    "fraud"
]

df = pd.DataFrame(
    data,
    columns=columns
)

# ----------------------------------
# Save Dataset
# ----------------------------------
df.to_csv(
    "data/invoices.csv",
    index=False
)

# ----------------------------------
# Summary
# ----------------------------------
print("\nDataset Created Successfully!")
print(f"Total Records: {len(df)}")

print("\nFraud Distribution:")
print(df["fraud"].value_counts())

print("\nSample Records:")
print(df.head())