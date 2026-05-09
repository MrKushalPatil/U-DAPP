from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
import plotly.express as px
import os
import numpy as np


def generate_pdf_report(df):

    os.makedirs("outputs/reports", exist_ok=True)
    file_path = "outputs/reports/eda_report.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    elements = []

    # ======================
    # TITLE
    # ======================
    elements.append(Paragraph("UDAP Pro - Advanced EDA Report", styles['Title']))
    elements.append(Spacer(1, 10))

    # ======================
    # BASIC INFO
    # ======================
    elements.append(Paragraph(f"Shape: {df.shape}", styles['Normal']))
    elements.append(Spacer(1, 10))

    # ======================
    # DATA TYPES
    # ======================
    dtype_df = df.dtypes.astype(str).reset_index()
    dtype_df.columns = ["Column", "Type"]

    table = [dtype_df.columns.tolist()] + dtype_df.values.tolist()
    elements.append(Table(table))
    elements.append(Spacer(1, 10))

    # ======================
    # MISSING VALUES
    # ======================
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column", "Missing"]

    table = [missing_df.columns.tolist()] + missing_df.values.tolist()
    elements.append(Table(table))
    elements.append(Spacer(1, 10))

    # ======================
    # SKEWNESS (NEW)
    # ======================
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:
        skew_df = pd.DataFrame({
            "Column": numeric_cols,
            "Skewness": df[numeric_cols].skew().values
        })

        table = [skew_df.columns.tolist()] + skew_df.values.tolist()
        elements.append(Paragraph("Skewness Analysis", styles['Heading3']))
        elements.append(Table(table))
        elements.append(Spacer(1, 10))

    # ======================
    # CONSTANT COLUMNS
    # ======================
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]

    elements.append(Paragraph(
        f"Constant Columns: {constant_cols if constant_cols else 'None'}",
        styles['Normal']
    ))
    elements.append(Spacer(1, 10))

    # ======================
    # HISTOGRAM
    # ======================
    if len(numeric_cols) > 0:
        col = numeric_cols[0]

        fig = px.histogram(df, x=col)
        img_path = f"outputs/reports/{col}_hist.png"
        fig.write_image(img_path)

        elements.append(Paragraph("Distribution", styles['Heading3']))
        elements.append(Image(img_path, width=400, height=250))
        elements.append(Spacer(1, 10))

    # ======================
    # CORRELATION
    # ======================
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()

        fig = px.imshow(corr, text_auto=True)
        img_path = "outputs/reports/corr.png"
        fig.write_image(img_path)

        elements.append(Paragraph("Correlation Matrix", styles['Heading3']))
        elements.append(Image(img_path, width=400, height=300))

    # ======================
    # BUILD
    # ======================
    doc.build(elements)

    return file_path