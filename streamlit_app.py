import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

        # Visualization
        fig, ax = plt.subplots()
        sns.scatterplot(x=input_data[:, 0], y=input_data[:, 2], hue=prediction, ax=ax, palette="deep")
        ax.set_xlabel('Sepal Length')
        ax.set_ylabel('Petal Length')
        st.pyplot(fig)

# Regression Analysis
elif analysis_type == "Regression":
    st.header("Regression Analysis with LinearRegression")
    st.write("Provide input features for regression prediction:")

    # Input features for regression
    rm = st.number_input('Number of Rooms (RM)', 3.0, 9.0, 6.0)
    lstat = st.number_input('Lower Status Population (LSTAT)', 1.0, 40.0, 12.0)
    ptratio = st.number_input('Pupil-Teacher Ratio (PTRATIO)', 12.0, 22.0, 18.0)

    input_data = np.array([[rm, lstat, ptratio]])

    if st.button('Predict Regression'):
        prediction = reg_model.predict(input_data)
        st.write(f'Predicted Value: {prediction[0]:.2f}')

        # Visualization
        fig, ax = plt.subplots()
        sns.scatterplot(x=input_data[:, 0], y=prediction, ax=ax, palette="deep")
        ax.set_xlabel('Number of Rooms (RM)')
        ax.set_ylabel('Predicted Value')
        st.pyplot(fig)

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

        # Visualization
        fig, ax = plt.subplots()
        sns.scatterplot(x=input_data[:, 0], y=input_data[:, 2], hue=prediction, ax=ax, palette="deep")
        ax.set_xlabel('Sepal Length')
        ax.set_ylabel('Petal Length')
        st.pyplot(fig)
