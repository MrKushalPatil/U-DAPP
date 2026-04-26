import pandas as pd
import streamlit as st
import io


# 🔹 Load Dataset (KEEP from your report.py, slightly improved)
def load_dataset(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# 🔹 Download Button
def download_csv(df, filename="processed_data.csv"):
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Processed Data",
        data=csv,
        file_name=filename,
        mime='text/csv'
    )


# 🔹 Basic Info (from your EDA logic)
def basic_info(df):
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }