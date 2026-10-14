"""utility functions for loading datasets."""

import pandas as pd

def load_data(uploaded_file):
    """loads a csv or excel file into a pandas dataframe."""
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)