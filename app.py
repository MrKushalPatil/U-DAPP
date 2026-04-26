import streamlit as st

from utils.helpers import load_dataset, download_csv

# Backend imports
from backend.column_classifier import detect_column_roles
from backend.preprocessing import (
    basic_cleaning,
    apply_formats,
    preprocess
)

from backend.eda import detailed_eda, data_quality_score
from backend.visualization import visualization_dashboard
from backend.ml_engine import auto_train_models
from backend.report_generator import generate_pdf_report


# =========================
# 🔹 PAGE CONFIG
# =========================
st.set_page_config(page_title="UDAPP", layout="wide")

# 🔹 Load CSS
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Universal Data Automation and Preprocessing Platform")

# =========================
# 🔹 SIDEBAR NAVIGATION
# =========================
menu = st.sidebar.radio(
    "Navigation",
    ["Upload Data", "Preprocessing", "EDA Report", "Visualization"]
)


# =========================
# 🔹 SESSION STATE
# =========================
if "df" not in st.session_state:
    st.session_state.df = None

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None


# =========================
# 📂 UPLOAD DATA
# =========================
if menu == "Upload Data":

    file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

    if file:
        df = load_dataset(file)
        st.session_state.df = df

        st.subheader("📄 Raw Data Preview")
        st.dataframe(df.head())


# =========================
# ⚙️ PREPROCESSING
# =========================
elif menu == "Preprocessing":

    if st.session_state.df is None:
        st.warning("Upload data first")
    else:
        df = st.session_state.df.copy()

        st.subheader("📄 Original Dataset")
        st.dataframe(df.head())

        # 🔹 Column Detection
        st.subheader("🧠 Column Detection")

        roles, confidences = detect_column_roles(df)

        # 🔹 Basic Cleaning
        st.subheader("🧹 Basic Cleaning (Auto Applied)")

        df_clean = basic_cleaning(df.copy(), roles)

        st.success("✅ Cleaning Completed")
        st.dataframe(df_clean.head())

        df = df_clean

        # 🔹 Edit Roles
        st.subheader("✏️ Edit Column Types")

        updated_roles = {}

        role_options = ["id","name","numeric","category","text","date","boolean","gender","count"]

        for col in df.columns:
            updated_roles[col] = st.selectbox(
                f"{col} (Confidence: {confidences[col]})",
                role_options,
                index=role_options.index(roles[col])
            )

        # 🔹 Select Columns
        st.subheader("🎯 Select Columns for Processing")

        selected_cols = st.multiselect("Choose columns", df.columns)

        # 🔹 Processing Options
        if selected_cols:

            st.subheader("⚙️ Processing Options")

            choices = {}

            for col in selected_cols:
                role = updated_roles[col]

                if role == "gender":
                    c = st.selectbox(f"{col} Gender Format", ["Male/Female","M/F","1/0"])
                    choices[col] = ("gender", str(["Male/Female","M/F","1/0"].index(c)+1))

                elif role == "boolean":
                    c = st.selectbox(f"{col} Boolean Format", ["1/0","True/False","Yes/No"])
                    choices[col] = ("boolean", str(["1/0","True/False","Yes/No"].index(c)+1))

                elif role == "date":
                    c = st.selectbox(
                        f"{col} Date Format",
                        ["yyyy-mm-dd","yyyy-dd-mm","dd-mm-yyyy","mm-dd-yyyy","Split"]
                    )
                    choices[col] = ("date", str(["yyyy-mm-dd","yyyy-dd-mm","dd-mm-yyyy","mm-dd-yyyy","Split"].index(c)+1))

                elif role == "category":
                    c = st.selectbox(
                        f"{col} Category Handling",
                        ["Keep","Encode 0/1","Encode True/False"]
                    )
                    choices[col] = ("category", str(["Keep","Encode 0/1","Encode True/False"].index(c)+1))

                elif role == "numeric":
                    c = st.selectbox(f"{col} Scaling", ["No","Yes"])
                    choices[col] = ("numeric", "1" if c == "Yes" else "2")

                elif role == "count":
                    choices[col] = ("count", "fixed")

            # 🔹 Apply Processing
            if st.button("🚀 Apply Processing"):

                df_processed = df.copy()

                selected_roles = {col: updated_roles[col] for col in selected_cols}

                df_processed = apply_formats(df_processed, selected_roles, choices)
                df_processed = preprocess(df_processed, selected_roles, choices)

                st.session_state.processed_df = df_processed

                st.success("✅ Processing Applied")

                st.subheader("📊 Final Dataset")
                st.dataframe(df_processed.head())

                download_csv(df_processed)


# =========================
# 📊 EDA REPORT
# =========================
elif menu == "EDA Report":

    if st.session_state.df is None:
        st.warning("Upload data first")
    else:
        data_option = st.radio("Select Data Source", ["Raw Data", "Processed Data"])

        df = st.session_state.df if data_option == "Raw Data" else st.session_state.processed_df

        if df is None:
            st.warning("Processed data not available")
        else:
            st.metric("Data Quality Score", data_quality_score(df))

            # 🔥 Full EDA
            detailed_eda(df)


# =========================
# 📊 VISUALIZATION
# =========================
elif menu == "Visualization":

    if st.session_state.df is None:
        st.warning("Upload data first")
    else:
        data_option = st.radio("Select Data Source", ["Raw Data", "Processed Data"])

        df = st.session_state.df if data_option == "Raw Data" else st.session_state.processed_df

        if df is None:
            st.warning("Processed data not available")
        else:
            visualization_dashboard(df)

