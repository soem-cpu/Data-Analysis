import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import base64
from io import BytesIO

# ========== Helper Functions ==========
def img_to_bytes(img_path):
    """Convert image to base64"""
    try:
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        return None

@st.cache_data
def load_data(file):
    """Load data from CSV or Excel file."""
    try:
        df = pd.read_csv(file)
        return df
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty.")
        return None
    except Exception as e:
        try:
            df = pd.read_excel(file)
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}. Please ensure the file is a valid CSV or Excel file.")
            return None

# ========== Page Configuration ==========
st.set_page_config(
    page_title="RESEARCH DATA ANALYSIS",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Custom CSS ==========
st.markdown("""
    <style>
        .stApp {
            background-color: #CADCAE;
        }
        .reportview-container .main .block-container {
            padding: 2rem;
            background-color: #ffffff;
            border-radius: 8px;
        }
        .sidebar .sidebar-content {
            background-color: #e0e0e0;
        }
    </style>
""", unsafe_allow_html=True)

# ========== Title Header with Image ==========
image_path = "chart icon01.png"
image_base64 = img_to_bytes(image_path)

if image_base64:
    st.markdown(f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{image_base64}" alt="Chart Icon" style="height: 50px; margin-right: 10px;">
            <h1>RESEARCH DATA ANALYSIS</h1>
        </div>
    """, unsafe_allow_html=True)
else:
    st.title("RESEARCH DATA ANALYSIS")
    st.warning(f"Image '{image_path}' not found. Ensure it is in the correct directory.")

st.header("Explore Your Data with Interactive Charts and Analysis")

# ========== Sidebar ==========
with st.sidebar:
    st.header("Settings")
    show_data = st.checkbox("Show Data Preview", True)
    show_descriptive_stats = st.checkbox("Show Descriptive Statistics", True)

    st.header("Chart Settings")
    selected_charts = st.multiselect(
        "Select Chart Types",
        ["Bar Chart", "Pie Chart", "Sankey Diagram",
         "100% Stacked Bar Chart", "Stacked Vertical Bar Chart", "Line Chart"]
    )

# ========== File Upload ==========
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    data = load_data(uploaded_file)

    if data is not None:
        # ========== Data Info ==========
        st.subheader("Data Information")
        st.write("Data Types:")
        st.write(data.dtypes)

        non_numeric_cols = data.select_dtypes(exclude=["number"]).columns.tolist()
        st.write("Non-numeric columns:", non_numeric_cols)

        # ========== Display Data and Statistics based on Sidebar Settings ==========
        if show_data:
            st.subheader("Data Preview")
            st.dataframe(data.head())

        if show_descriptive_stats:
            st.subheader("Descriptive Statistics")
            st.dataframe(data.describe())

        # ========== Numeric-only Data ==========
        data_numeric = data.drop(non_numeric_cols, axis=1, errors='ignore').fillna(0)

        # ========== Correlation Heatmap ==========
        st.subheader("Correlation Heatmap")
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(data_numeric.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Failed to plot heatmap: {e}")

        # ========== Histograms ==========
        st.subheader("Histograms")
        hist_columns = st.multiselect("Select columns for histogram", data_numeric.columns)
        for col in hist_columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(data[col], kde=True, ax=ax)
            st.pyplot(fig)

        # ========== Scatter Plots ==========
        st.subheader("Scatter Plot")
        x_col = st.selectbox("X-axis", data_numeric.columns)
        y_col = st.selectbox("Y-axis", data_numeric.columns, index=1)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=data[x_col], y=data[y_col], ax=ax)
        st.pyplot(fig)

        # ========== Categorical Columns ==========
        st.subheader("Value Counts (Categorical Columns)")
        for col in data.select_dtypes(include=['object']).columns:
            st.markdown(f"**{col}**")
            st.dataframe(data[col].value_counts())

        # ========== Machine Learning Model ==========
        st.subheader("Simple Machine Learning (Logistic Regression)")
        if 'target' in data.columns:
            X = data.drop('target', axis=1, errors='ignore')
            y = data['target']

            # Ensure X only contains numeric columns and handle missing values
            X = X.select_dtypes(include=['number']).fillna(0)
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(X.mean(), inplace=True)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.markdown(f"**Accuracy:** `{acc:.2f}`")
        else:
            st.warning("Column 'target' not found for ML model.")

        # ========== Additional Charts ==========
        st.subheader("Additional Charts")
        for chart in selected_charts:
            st.markdown(f"### {chart}")
            if chart == "Bar Chart":
                x = st.selectbox(f"{chart} - X axis", data.columns, key=f"{chart}_x")
                y = st.selectbox(f"{chart} - Y axis", data.columns, key=f"{chart}_y")
                st.plotly_chart(px.bar(data, x=x, y=y))

            elif chart == "Pie Chart":
                col = st.selectbox(f"{chart} - Column", data.columns, key=f"{chart}_col")
                st.plotly_chart(px.pie(data, names=col))

            elif chart == "Sankey Diagram":
                source = st.selectbox("Source column", data.columns, key="sankey_src")
                target = st.selectbox("Target column", data.columns, key="sankey_trg")
                value = st.selectbox("Value column", data.columns, key="sankey_val")
                if all([source, target, value]):
                    fig = px.sunburst(data, path=[source, target], values=value)
                    st.plotly_chart(fig)

            elif chart == "100% Stacked Bar Chart":
                x = st.selectbox("X axis", data.columns, key="stacked100_x")
                y = st.selectbox("Y axis", data.columns, key="stacked100_y")
                color = st.selectbox("Color (group by)", data.columns, key="stacked100_color")
                grouped = data.groupby([x, color])[y].sum().unstack().fillna(0)
                grouped_percent = grouped.div(grouped.sum(axis=1), axis=0) * 100
                st.plotly_chart(px.bar(grouped_percent, x=grouped_percent.index, y=grouped_percent.columns,
                                       labels={"value": "Percentage"}))

            elif chart == "Stacked Vertical Bar Chart":
                x = st.selectbox("X axis", data.columns, key="stacked_x")
                y = st.selectbox("Y axis", data.columns, key="stacked_y")
                color = st.selectbox("Color", data.columns, key="stacked_color")
                st.plotly_chart(px.bar(data, x=x, y=y, color=color))

            elif chart == "Line Chart":
                x = st.selectbox("X axis", data.columns, key="line_x")
                y = st.selectbox("Y axis", data.columns, key="line_y")
                st.plotly_chart(px.line(data, x=x, y=y))
else:
    st.info("Please upload a CSV or Excel file to begin.")
