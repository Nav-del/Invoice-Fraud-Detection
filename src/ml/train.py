import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/invoices.csv")

# -----------------------------
# Encode Categorical Columns
# -----------------------------
categorical_columns = [
    "vendor_name",
    "gst_number",
    "invoice_category",
    "payment_method"
]

encoders = {}

for col in categorical_columns:
    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(df[col])

    encoders[col] = encoder

# Save encoders
joblib.dump(
    encoders,
    "models/encoders.pkl"
)

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop(
    columns=[
        "invoice_number",
        "invoice_date",
        "due_date",
        "fraud"
    ]
)

y = df["fraud"]

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Random Forest
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
print("\nAccuracy:")
print(
    accuracy_score(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(
    model,
    "models/fraud_model.pkl"
)

print("\nModel Saved!")