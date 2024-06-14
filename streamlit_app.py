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

# Classification Analysis
if analysis_type == "Classification":
    st.header("Classification Analysis with RandomForestClassifier")
    st.write("Provide input features for classification prediction:")

    # Input features for classification
    sepal_length = st.number_input('Sepal Length', 4.0, 8.0, 5.5)
    sepal_width = st.number_input('Sepal Width', 2.0, 4.5, 3.0)
    petal_length = st.number_input('Petal Length', 1.0, 7.0, 4.0)
    petal_width = st.number_input('Petal Width', 0.1, 2.5, 1.0)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button('Predict Classification'):
        prediction = clf_model.predict(input_data)
        st.write(f'Predicted Class: {prediction[0]}')

# Regression Analysis
elif analysis_type == "Regression":
    st.header("Regression Analysis with LinearRegression")
    st.write("Provide input features for regression prediction:")

    # Input features for regression
    medinc = st.number_input('MedInc', 0.0, 15.0, 3.0)
    houseage = st.number_input('HouseAge', 0.0, 52.0, 15.0)
    ave_rooms = st.number_input('AveRooms', 0.0, 10.0, 5.0)
    ave_bedrms = st.number_input('AveBedrms', 0.0, 10.0, 1.0)
    population = st.number_input('Population', 0.0, 35000.0, 1500.0)
    ave_occup = st.number_input('AveOccup', 0.0, 50.0, 3.0)
    latitude = st.number_input('Latitude', 32.0, 42.0, 35.0)
    longitude = st.number_input('Longitude', -125.0, -114.0, -120.0)

    input_data = np.array([[medinc, houseage, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])

    if st.button('Predict Regression'):
        prediction = reg_model.predict(input_data)
        st.write(f'Predicted Value: {prediction[0]:.2f}')

# Clustering Analysis
elif analysis_type == "Clustering":
    st.header("Clustering Analysis with KMeans")
    st.write("Provide input features for clustering:")

    # Input features for clustering
    sepal_length = st.number_input('Sepal Length', 4.0, 8.0, 5.5)
    sepal_width = st.number_input('Sepal Width', 2.0, 4.5, 3.0)
    petal_length = st.number_input('Petal Length', 1.0, 7.0, 4.0)
    petal_width = st.number_input('Petal Width', 0.1, 2.5, 1.0)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button('Predict Cluster'):
        prediction = kmeans_model.predict(input_data)
        st.write(f'Predicted Cluster: {prediction[0]}')
