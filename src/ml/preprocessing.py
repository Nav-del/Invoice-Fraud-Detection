'''
Data Preprocessing

Prepares the featured dataset for machine learning
'''

import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

#
# Load Dataset
#

'''
Reads the featured_dataset.csv into a Pandas dataframe so all operations can be applied to it.
Instead of reading the CVS repeateadly, we load it once and pass the DataFrame through the pipeline


Keeping dataset loading separate makes the pipeline modular. If later the dataset path changes, we only update this one function.
'''
def load_dataset(path):
    print("Loading the featured dataset...")

    df=pd.read_csv(path)

    return df

#
# Drop Unnecessary Columns
#
'''
Identifier columns uniquely identify an invoice but dont contain meaningful fraud information.
Similarly, we've already extracted all useful information from the date columns into engineered features.
Removing these columns prevents the model from memorizing IDs and avoids feeding duplicate information into the training process.
'''

def drop_unnecessary_columns(df):

    print("Dropping Identifier Columns...")

    columns_to_drop=["invoice_number","purchase_order","vendor_id","invoice_date","due_date"]

    df=df.drop(columns = columns_to_drop)

    return df

#
# Handle Missing values
#
'''
Machine learning models cannot train on datasets with missing values. This function identifies numerical and categorical columns separately and fills missing values using an appropriate strategy.

Numerical columns are filled with the median, which is less affected by extreme values than the mean.
Categorical columns are filled with the most frequent value, preserving the existing data distribution without introducing new categories.
'''

def handle_missing_values(df):

    print("Handling Missing Values...")

    numerical_columns = df.select_dtypes(include=["number"]).columns

    categorical_columns = df.select_dtypes(include=["object"]).columns

    numerical_imputer = SimpleImputer(strategy="median")

    categorical_imputer = SimpleImputer(strategy="most_frequent")

    df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])

    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    return df

#
# Handle Categorical Columns
#
'''
Only the columns that represent meaningful categories are encoded. The GST number is intentionally excluded because it is not a categorical feature in the traditional sense; it is an identifier that may later be transformed into richer validation features.

Using unknown_value=-1 ensures that if a new vendor, category, or payment method appears during prediction, the pipeline won't crash—it will encode unseen categories as -1.
'''

def encode_categorical_columns(df):

    print("Encoding categorical columns...")

    categorical_columns = ["vendor_name","category","country","state","currency","payment_method","payment_status","invoide_description","vendor_risk","risk_severity"]

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)

    df[categorical_columns] = encoder.fit_transform(df[categorical_columns])

    return df, encoder

#
#Save encoder 
#
'''
The fitted OrdinalEncoder contains the learned mapping between each category and its encoded numerical value. Saving it ensures that the exact same mapping is reused during prediction.

Consistency between training and inference is essential. If categories are encoded differently at prediction time, the model will receive incorrect feature values and produce unreliable predictions.
'''

def save_encoder(encoder, path):

    print("Saving Encoder...")

    joblib.dump(encoder, path)


#
# Save feature columns
#
'''
The machine learning model expects input features in the same order used during training. Saving the feature list preserves this order so future invoices can be transformed correctly before prediction.
'''

def save_feature_columns(df, path):

    print("Saving Feature Columns...")

    feature_columns = df.columns.tolist()

    joblib.dump(feature_columns, path)