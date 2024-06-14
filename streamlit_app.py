import streamlit as st
import joblib
import numpy as np

# Load models
clf_model = joblib.load('clf_model.pkl')
reg_model = joblib.load('reg_model.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')

# Page title
st.title("Real-Time Analysis: Classification, Regression, and Clustering")

# Sidebar for user inputs
st.sidebar.title("Choose Analysis Type")
analysis_type = st.sidebar.selectbox("Analysis Type", ["Classification", "Regression", "Clustering"])

# Helper function to parse input data
def parse_input(input_text):
    data = []
    for line in input_text.strip().split('\n'):
        data.append([float(x) for x in line.split(',')])
    return np.array(data)

# Classification Analysis
if analysis_type == "Classification":
    st.header("Classification Analysis with RandomForestClassifier")
    st.write("Provide input features for classification prediction (one row per line, values separated by commas):")

    # Input features for classification
    input_text = st.text_area('Input Features', '5.1,3.5,1.4,0.2\n4.9,3.0,1.4,0.2')

    if st.button('Predict Classification'):
        input_data = parse_input(input_text)
        predictions = clf_model.predict(input_data)
        st.write('Predicted Classes:', predictions)

# Regression Analysis
elif analysis_type == "Regression":
    st.header("Regression Analysis with LinearRegression")
    st.write("Provide input features for regression prediction (one row per line, values separated by commas):")

    # Input features for regression
    input_text = st.text_area('Input Features', '3.0,15.0,5.0,1.0,1500.0,3.0,35.0,-120.0\n4.0,20.0,6.0,2.0,2000.0,4.0,36.0,-121.0')

    if st.button('Predict Regression'):
        input_data = parse_input(input_text)
        predictions = reg_model.predict(input_data)
        st.write('Predicted Values:', predictions)

# Clustering Analysis
elif analysis_type == "Clustering":
    st.header("Clustering Analysis with KMeans")
    st.write("Provide input features for clustering (one row per line, values separated by commas):")

    # Input features for clustering
    input_text = st.text_area('Input Features', '5.1,3.5,1.4,0.2\n4.9,3.0,1.4,0.2')

    if st.button('Predict Cluster'):
        input_data = parse_input(input_text)
        predictions = kmeans_model.predict(input_data)
        st.write('Predicted Clusters:', predictions)
