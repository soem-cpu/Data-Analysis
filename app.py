import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # For interactive charts like sunkey and more
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import Image
import base64

# Helper function to convert image to base64
def img_to_bytes(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Set page configuration
st.set_page_config(
    page_title="RESEARCH DATA ANALYSIS",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for theme
st.markdown("""
    <style>
        body {
            color: #000; /* Black text */
            background-color: #CADCAE; /* Light green background */
        }
        .stApp {
            background-color: #CADCAE; /* Light greeen background */
        }
        .st-bb { /* Streamlit main content */
            background-color: #fff; /* White background for content area */
            padding: 20px;
            border-radius: 5px;
        }
        .st-at { /* Streamlit sidebar */
            background-color: #e0e0e0; /* Light gray sidebar background */
        }
    </style>
    """, unsafe_allow_html=True)

# Load the image
image_path = "chart icon01.png"
try:
    image_base64 = img_to_bytes(image_path)
    image_html = f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{image_base64}" alt="Chart Icon" style="margin-right: 10px; height: 50px;">
            <h1 style="display: inline;">RESEARCH DATA ANALYSIS</h1>
        </div>
        """
    st.markdown(image_html, unsafe_allow_html=True)
except FileNotFoundError:
    st.warning(f"Image file '{image_path}' not found. Please make sure it is in the same directory as the script.")
    st.title("RESEARCH DATA ANALYSIS")

st.header("Explore Your Data with Interactive Charts and Analysis")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    show_data = st.checkbox("Show Data Preview", value=True)
    show_descriptive_stats = st.checkbox("Show Descriptive Statistics", value=True)
    st.header("Chart Settings")
    selected_charts = st.multiselect(
        "Select chart types",
        ["Bar Chart", "Pie Chart", "Sunkey Diagram",
         "100% Stacked Bar Chart", "Stacked Vertical Bar Chart",
         "Line Chart"]
    )

# Load the dataset
@st.cache_data  # Cache the data to avoid reloading on every interaction
def load_data(file_path, file_type):
    if file_type == "csv":
        data = pd.read_csv(file_path)
    elif file_type == "xlsx":
        data = pd.read_excel(file_path)
    return data

# File uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    data = load_data(uploaded_file, file_type)

    # Display data preview
    if show_data:
        st.markdown("### Data Preview")
        st.dataframe(data.head())

    # Descriptive Statistics
    if show_descriptive_stats:
        st.markdown("### Descriptive Statistics")
        st.dataframe(data.describe())

    # Data Visualization
    st.markdown("### Data Visualization")

    # Data Types
    st.markdown("### Data Types")
    st.write(data.dtypes)

    # Handle non-numeric columns
    non_numeric_cols = data.select_dtypes(exclude=['number']).columns
    st.write("Non-numeric columns:", non_numeric_cols)

    # Remove non-numeric columns
    data_numeric = data.drop(non_numeric_cols, axis=1, errors='ignore')

    # Handle missing values
    data_numeric = data_numeric.fillna(0)

    # Correlation heatmap
    st.markdown("### Correlation Heatmap")
    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(data_numeric.corr(), annot=True, cmap="coolwarm", linewidths=.5, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.write(f"Error creating heatmap: {e}")

    # Choose columns for histogram
    st.markdown("### Histograms")
    hist_columns = st.multiselect("Select columns for histogram", data.columns, key="hist_columns")

    if hist_columns:
        for column in hist_columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(data[column], kde=True, ax=ax)
            st.pyplot(fig)

    # Choose columns for scatter plot
    st.markdown("### Scatter Plots")
    x_column = st.selectbox("Select X axis", data.columns, key="x_column")
    y_column = st.selectbox("Select Y axis", data.columns, key="y_column")

    if x_column and y_column:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data, x=x_column, y=y_column, ax=ax)
        st.pyplot(fig)

    # Basic Data Analysis
    st.markdown("### Basic Data Analysis")

    # Display value counts for categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    for column in categorical_cols:
        st.markdown(f"#### Value Counts for {column}")
        st.dataframe(data[column].value_counts())

    # Simple Machine Learning Model
    st.markdown("### Simple Machine Learning Model")

    # Prepare data for machine learning
    if 'target' in data.columns:
        X = data.drop('target', axis=1, errors='ignore')
        y = data['target']

        # Ensure numeric data types in X
        X = X.select_dtypes(include=['number'])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Handle missing values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        # Remove infinite values
        X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill remaining NaN values with the mean
        X_train.fillna(X_train.mean(), inplace=True)
        X_test.fillna(X_test.mean(), inplace=True)

        # Train a Logistic Regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Make predictions and evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.markdown(f"#### Accuracy: {accuracy}")
    else:
        st.write("Target column 'target' not found in the dataset.")

    # Additional Charts
    st.markdown("### Additional Charts")

    for chart_type in selected_charts:
        st.markdown(f"### {chart_type}")  # Display the chart type as a header

        if chart_type == "Bar Chart":
            column_x = st.selectbox("Select X axis for Bar Chart", data.columns, key=f"bar_x_{chart_type}")
            column_y = st.selectbox("Select Y axis for Bar Chart", data.columns, key=f"bar_y_{chart_type}")
            if column_x and column_y:
                fig = px.bar(data, x=column_x, y=column_y)
                st.plotly_chart(fig)

        elif chart_type == "Pie Chart":
            column = st.selectbox("Select column for Pie Chart", data.columns, key=f"pie_{chart_type}")
            if column:
                fig = px.pie(data, names=column)
                st.plotly_chart(fig)

        elif chart_type == "Sunkey Diagram":
            source = st.selectbox("Select source column for Sunkey Diagram", data.columns, key=f"sunkey_source_{chart_type}")
            target = st.selectbox("Select target column for Sunkey Diagram", data.columns, key=f"sunkey_target_{chart_type}")
            values = st.selectbox("Select values column for Sunkey Diagram", data.columns, key=f"sunkey_values_{chart_type}")
            if source and target and values:
                fig = px.sunburst(data, path=[source, target], values=values)
                st.plotly_chart(fig)

        elif chart_type == "100% Stacked Bar Chart":
            column_x = st.selectbox("Select X axis for 100% Stacked Bar Chart", data.columns, key=f"stacked100_x_{chart_type}")
            column_y = st.selectbox("Select Y axis for 100% Stacked Bar Chart", data.columns, key=f"stacked100_y_{chart_type}")
            color = st.selectbox("Select color column for 100% Stacked Bar Chart", data.columns, key=f"stacked100_color_{chart_type}")
            if column_x and column_y and color:
                # Group data and calculate percentages
                grouped = data.groupby([column_x, color])[column_y].sum().unstack().fillna(0)
                grouped = grouped.div(grouped.sum(axis=1), axis=0) * 100
                fig = px.bar(grouped, x=grouped.index, y=grouped.columns,
                             labels={'value': 'Percentage'},
                             title='100% Stacked Bar Chart')
                st.plotly_chart(fig)

        elif chart_type == "Stacked Vertical Bar Chart":
            column_x = st.selectbox("Select X axis for Stacked Vertical Bar Chart", data.columns, key=f"vertstacked_x_{chart_type}")
            column_y = st.selectbox("Select Y axis for Stacked Vertical Bar Chart", data.columns, key=f"vertstacked_y_{chart_type}")
            color = st.selectbox("Select color column for Stacked Vertical Bar Chart", data.columns, key=f"vertstacked_color_{chart_type}")
            if column_x and column_y and color:
                fig = px.bar(data, x=column_x, y=column_y, color=color,
                             title='Stacked Vertical Bar Chart')
                st.plotly_chart(fig)

        elif chart_type == "Line Chart":
            column_x = st.selectbox("Select X axis for Line Chart", data.columns, key=f"line_x_{chart_type}")
            column_y = st.selectbox("Select Y axis for Line Chart", data.columns, key=f"line_y_{chart_type}")
            if column_x and column_y:
                fig = px.line(data, x=column_x, y=column_y,
                              title='Line Chart')
                st.plotly_chart(fig)

else:
    st.warning("Please upload a CSV or Excel file to proceed.")
