"""utility functions for loading and cleaning datasets."""

import pandas as pd

def load_data(uploaded_file):
    """loads a csv or excel file into a pandas dataframe."""
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

def clean_data(df, action, column_name=None):
    """performs data cleaning operations on the dataframe."""
    cleaned_df = df.copy()
    
    if action == "Drop Duplicates":
        cleaned_df = cleaned_df.drop_duplicates()
    elif action == "Drop Rows with Missing Values":
        cleaned_df = cleaned_df.dropna()
    elif action == "Fill Missing Numeric Values (Mean)" and column_name:
        if column_name in cleaned_df.columns:
            mean_val = cleaned_df[column_name].mean()
            cleaned_df[column_name] = cleaned_df[column_name].fillna(mean_val)
            
    return cleaned_df