import os
import random
import pandas as pd
from datetime import datetime, timedelta

# -----------------------------------------
# Configuration
# -----------------------------------------

NUM_VENDORS = 100

OUTPUT_PATH = "data/raw/vendors.csv"

os.makedirs("data/raw", exist_ok=True)

# -----------------------------------------
# Master Data
# -----------------------------------------

vendor_prefixes = [
    "ABC", "Tech", "Global", "Prime", "Future",
    "Vertex", "Skyline", "Bluechip", "Apex",
    "Quantum", "Nova", "Vision", "Delta",
    "Alpha", "Sigma", "Zenith"
]

vendor_suffixes = [
    "Technologies",
    "Solutions",
    "Systems",
    "Electronics",
    "Consulting",
    "Private Limited",
    "Enterprises",
    "Industries",
    "Services",
    "Software",
    "Networks",
    "Corporation"
]

categories = {
    "Software": (20000, 80000),
    "Hardware": (80000, 300000),
    "Electronics": (15000, 150000),
    "Consulting": (50000, 200000),
    "Maintenance": (10000, 60000),
    "Cloud Services": (30000, 120000),
    "Networking": (25000, 180000),
    "Office Supplies": (5000, 40000)
}

states = {
    "KA": "29",
    "MH": "27",
    "TN": "33",
    "DL": "07",
    "GJ": "24",
    "TS": "36",
    "WB": "19",
    "RJ": "08"
}

payment_methods = [
    "Bank Transfer",
    "NEFT",
    "RTGS",
    "UPI",
    "Cheque"
]

risk_levels = [
    "Low",
    "Medium",
    "High"
]

# -----------------------------------------
# GST Generator
# -----------------------------------------

def generate_gst(state_code):

    pan = (
        ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
        + ''.join(random.choices("0123456789", k=4))
        + random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )

    return (
        state_code
        + pan
        + random.choice("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        + "Z"
        + random.choice("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )

# -----------------------------------------
# Vendor Name Generator
# -----------------------------------------

def generate_vendor_name():

    return (
        random.choice(vendor_prefixes)
        + " "
        + random.choice(vendor_suffixes)
    )

# -----------------------------------------
# Generate Vendors
# -----------------------------------------

vendors = []

used_names = set()

for i in range(NUM_VENDORS):

    while True:

        vendor_name = generate_vendor_name()

        if vendor_name not in used_names:

            used_names.add(vendor_name)

            break

    vendor_id = f"VEN{1000+i}"

    category = random.choice(list(categories.keys()))

    avg_invoice = random.randint(
        categories[category][0],
        categories[category][1]
    )

    state = random.choice(list(states.keys()))

    gst_number = generate_gst(states[state])

    invoice_frequency = random.randint(5, 150)

    payment_method = random.choice(payment_methods)

    vendor_since = (
        datetime.today()
        - timedelta(days=random.randint(365, 3650))
    ).date()

    # 5% blacklisted

    blacklisted = (
        random.random() < 0.05
    )

    # Risk Distribution

    r = random.random()

    if r < 0.70:

        risk = "Low"

    elif r < 0.90:

        risk = "Medium"

    else:

        risk = "High"

    vendors.append({

        "vendor_id": vendor_id,

        "vendor_name": vendor_name,

        "gst_number": gst_number,

        "category": category,

        "country": "India",

        "state": state,

        "average_invoice_amount": avg_invoice,

        "invoice_frequency": invoice_frequency,

        "preferred_payment": payment_method,

        "vendor_since": vendor_since,

        "blacklisted": blacklisted,

        "risk_level": risk

    })

# -----------------------------------------
# Save
# -----------------------------------------

df = pd.DataFrame(vendors)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nVendor Master Generated Successfully!\n")

print(df.head())

print(f"\nTotal Vendors : {len(df)}")

print(f"Saved to : {OUTPUT_PATH}")