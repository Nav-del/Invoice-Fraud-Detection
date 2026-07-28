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

