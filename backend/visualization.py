import streamlit as st
import plotly.express as px
import tempfile


def visualization_dashboard(df):

    st.subheader("📊 Advanced Visualization Dashboard")

    columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # ============================
    # STEP 1: SELECT COLUMNS
    # ============================
    st.subheader("🎯 Step 1: Select Columns")

    selected_columns = st.multiselect("Select Columns", columns)

    if not selected_columns:
        st.warning("Select at least one column")
        return

    # ============================
    # STEP 2: SELECT CHART
    # ============================
    st.subheader("📊 Step 2: Select Chart Type")

    chart_type = st.selectbox(
        "Chart Type",
        [
            "Histogram", "Boxplot", "Scatter", "Line", "Bar",
            "Pie", "Donut", "Area", "Violin",
            "Heatmap", "Density", "Sunburst", "Treemap"
        ]
    )

    fig = None

    # ============================
    # STEP 3: DYNAMIC PARAMETERS
    # ============================
    st.subheader("⚙️ Step 3: Configure Chart")

    x_col, y_col, color_col = None, None, None

    # 1-column charts
    if chart_type in ["Histogram", "Pie", "Donut"]:
        x_col = st.selectbox("Select Column", selected_columns)

    # 2-column charts
    elif chart_type in ["Scatter", "Line", "Bar", "Area", "Density"]:
        x_col = st.selectbox("X-axis", selected_columns)
        y_col = st.selectbox("Y-axis", selected_columns)

    # Special charts
    elif chart_type == "Boxplot":
        x_col = st.selectbox("Category (X)", selected_columns)
        y_col = st.selectbox("Value (Y)", selected_columns)

    elif chart_type == "Violin":
        x_col = st.selectbox("Category (X)", selected_columns)
        y_col = st.selectbox("Value (Y)", selected_columns)

    elif chart_type == "Heatmap":
        st.info("Uses all numeric columns automatically")

    elif chart_type in ["Sunburst", "Treemap"]:
        path_cols = st.multiselect("Hierarchy Columns", selected_columns)

    # Optional color
    if chart_type not in ["Heatmap"]:
        color_col = st.selectbox("Color (optional)", ["None"] + selected_columns)
        color_col = None if color_col == "None" else color_col

    # ============================
    # CHART CREATION
    # ============================

    if chart_type == "Histogram":
        fig = px.histogram(df, x=x_col, color=color_col)

    elif chart_type == "Boxplot":
        fig = px.box(df, x=x_col, y=y_col, color=color_col)

    elif chart_type == "Scatter":
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col)

    elif chart_type == "Line":
        fig = px.line(df, x=x_col, y=y_col, color=color_col)

    elif chart_type == "Bar":
        fig = px.bar(df, x=x_col, y=y_col, color=color_col)

    elif chart_type == "Pie":
        fig = px.pie(df, names=x_col)

    elif chart_type == "Donut":
        fig = px.pie(df, names=x_col, hole=0.4)

    elif chart_type == "Area":
        fig = px.area(df, x=x_col, y=y_col, color=color_col)

    elif chart_type == "Violin":
        fig = px.violin(df, x=x_col, y=y_col, color=color_col, box=True)

    elif chart_type == "Heatmap":
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns")
            return
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True)

    elif chart_type == "Density":
        fig = px.density_contour(df, x=x_col, y=y_col)

    elif chart_type == "Sunburst":
        if path_cols:
            fig = px.sunburst(df, path=path_cols)

    elif chart_type == "Treemap":
        if path_cols:
            fig = px.treemap(df, path=path_cols)

    # ============================
    # DISPLAY + DOWNLOAD
    # ============================
    if fig:
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📥 Download Visualization")

        # PNG
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.write_image(tmpfile.name)
            with open(tmpfile.name, "rb") as f:
                st.download_button("Download PNG", f, "chart.png")

        # HTML
        html_bytes = fig.to_html().encode("utf-8")
        st.download_button("Download Interactive HTML", html_bytes, "chart.html")

    else:
        st.warning("⚠️ Please configure chart properly")