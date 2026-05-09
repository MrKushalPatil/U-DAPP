import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from ydata_profiling import ProfileReport
import streamlit.components.v1 as components


# ================================
# DATA QUALITY SCORE
# ================================
def data_quality_score(df):
    missing = df.isnull().sum().sum()
    total = df.size
    completeness = (1 - (missing / total)) * 100
    return round(completeness, 2)


# ================================
# PROFILING-STYLE EDA
# ================================
def detailed_eda(df):

    st.subheader("📊 Advanced EDA (YData-like Profiling)")

    # =========================
    # BASIC INFO
    # =========================
    st.markdown("### 📄 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Quality Score", f"{data_quality_score(df)}%")

    st.write("Columns:", list(df.columns))

    # =========================
    # DATA TYPES
    # =========================
    st.markdown("### 🧬 Data Types")
    st.dataframe(df.dtypes.astype(str))

    # =========================
    # MISSING VALUES (IMPROVED)
    # =========================
    st.markdown("### 🧹 Missing Values Analysis")

    missing = df.isnull().sum()

    col1, col2 = st.columns(2)
    col1.metric("Total Missing", int(missing.sum()))
    col2.metric("Duplicate Rows", int(df.duplicated().sum()))

    missing_df = missing.reset_index()
    missing_df.columns = ["Column", "Missing"]

    fig = px.bar(missing_df, x="Column", y="Missing")
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # CONSTANT / LOW VARIANCE COLUMNS
    # =========================
    st.markdown("### ⚠️ Constant / Low-Variance Columns")

    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]

    if constant_cols:
        st.warning(f"Constant columns detected: {constant_cols}")
    else:
        st.success("No constant columns found")

    # =========================
    # STATISTICS
    # =========================
    st.markdown("### 📈 Statistical Summary")
    st.dataframe(df.describe(include='all'))

    # =========================
    # SKEWNESS (NEW LIKE PROFILING)
    # =========================
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:
        st.markdown("### 📉 Skewness Analysis")

        skew_df = pd.DataFrame({
            "Column": numeric_cols,
            "Skewness": df[numeric_cols].skew().values
        })

        st.dataframe(skew_df)

    # =========================
    # NUMERIC DISTRIBUTIONS
    # =========================
    if len(numeric_cols) > 0:
        st.markdown("### 📊 Distributions")

        selected_col = st.selectbox("Select Numeric Column", numeric_cols)

        fig1 = px.histogram(df, x=selected_col, marginal="box")
        st.plotly_chart(fig1, use_container_width=True)

    # =========================
    # CATEGORICAL ANALYSIS
    # =========================
    cat_cols = df.select_dtypes(include='object').columns

    if len(cat_cols) > 0:
        st.markdown("### 🧾 Categorical Analysis")

        selected_cat = st.selectbox("Select Categorical Column", cat_cols)

        value_counts = df[selected_cat].value_counts().reset_index()
        value_counts.columns = [selected_cat, "Count"]

        fig2 = px.bar(value_counts, x=selected_cat, y="Count")
        st.plotly_chart(fig2, use_container_width=True)

    # =========================
    # CORRELATION (IMPROVED)
    # =========================
    if len(numeric_cols) > 1:
        st.markdown("### 🔥 Correlation Heatmap")

        corr = df[numeric_cols].corr(method="pearson")

        fig3 = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig3, use_container_width=True)

    # =========================
    # FULL PROFILING REPORT (NEW)
    # =========================
    st.markdown("### 🧠 Full Auto Profiling Report ")

    if st.button("Generate Full Profiling Report"):

        profile = ProfileReport(df, explorative=True)

        profile.to_file("outputs/profile_report.html")

        st.success("Report generated!")

        HtmlFile = open("outputs/profile_report.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()

        components.html(source_code, height=1000, scrolling=True)