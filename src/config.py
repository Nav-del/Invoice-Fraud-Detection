"""
Project Configuration
Invoice Fraud Detection System
"""

# ================================
# DATASET
# ================================

NUM_VENDORS = 100
TOTAL_INVOICES = 10000
FRAUD_RATE = 0.30

# ================================
# GST
# ================================

GST_RATES = {
    "Software": 18,
    "Hardware": 18,
    "Consulting": 18,
    "Electronics": 28,
    "Office Supplies": 12,
    "Maintenance": 18,
    "Cloud Services": 18,
    "Networking": 18
}

# ================================
# PAYMENT
# ================================

PAYMENT_METHODS = [
    "Bank Transfer",
    "NEFT",
    "RTGS",
    "UPI",
    "Cheque"
]

PAYMENT_STATUS = [
    "Paid",
    "Pending",
    "Processing",
    "Overdue"
]

PAYMENT_TERMS = [
    15,
    30,
    45,
    60
]

# ================================
# CURRENCY
# ================================

CURRENCY = "INR"

# ================================
# DESCRIPTIONS
# ================================

DESCRIPTIONS = [
    "Software License Renewal",
    "Cloud Infrastructure Subscription",
    "Laptop Procurement",
    "Network Switch Installation",
    "Annual Maintenance Contract",
    "Server Upgrade",
    "Database Consulting",
    "Office Furniture Purchase",
    "Printer Cartridge Supply",
    "Cyber Security Audit",
    "Firewall Installation",
    "IT Infrastructure Upgrade",
    "Cloud Storage Expansion",
    "Desktop Computer Purchase",
    "Business Analytics Consulting"
]

# ================================
# RISK WEIGHTS
# ================================

RISK_WEIGHTS = {
    "duplicate_invoice": 30,
    "invalid_gst": 25,
    "tax_mismatch": 20,
    "date_anomaly": 15,
    "amount_anomaly": 20,
    "blacklisted_vendor": 35,
    "duplicate_po": 20,
    "payment_anomaly": 15
}