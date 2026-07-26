"""
Feature Builder

Runs all feature engineering modules
and creates the final featured dataset.
"""
import pandas as pd

from src.feature_engineering.financial_features import (create_financial_features)

from src.feature_engineering.temporal_features import (create_temporal_features)

from src.feature_engineering.risk_features import (create_risk_features)

#
# Main Function
#
'''
Explanation
Instead of calling every feature engineering function manually, this function executes all three feature engineering modules one after another in a fixed order.

This creates a single pipeline where the input is the original fraud dataset and the output is a fully engineered dataset ready for preprocessing and model training.
'''

def build_features(df):

    print("\nStarting Feature Engineering Pipeline : \n")

    df = create_financial_features(df)

    df = create_temporal_features(df)

    df = create_risk_features(df)

    print("\nFeature Engineering Completed!\n")

    return df

if __name__ == "__main__":

    df = pd.read_csv(
        "data/processed/fraud_dataset.csv"
    )

    df = build_features(df)

    output_path = (
        "data/processed/featured_dataset.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(f"\nDataset saved to: {output_path}")

    print("\nFinal Shape:", df.shape)