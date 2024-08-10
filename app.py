import streamlit as st
import numpy as np
import pickle
import os

# Function to load the model
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Load model and PCA transformer with error handling
model_path = os.path.join('models', 'best_model.pkl')
pca_path = os.path.join('models', 'pca_transformer.pkl')

try:
    model = load_model(model_path)
    pca = load_model(pca_path)
except Exception as e:
    st.error(f"Error loading model or PCA transformer: {e}")
    st.stop()

# Streamlit App
st.title('Predictive Maintenance Model')
st.write("This application predicts potential failures in equipment based on input features.")

st.header('Enter the input features:')

# Input fields for user
air_temp = st.number_input('Air Temperature [K]', value=298.1)
process_temp = st.number_input('Process Temperature [K]', value=308.6)
rot_speed = st.number_input('Rotational Speed [rpm]', value=1551)
torque = st.number_input('Torque [Nm]', value=42.8)
tool_wear = st.number_input('Tool Wear [min]', value=0.0)

# Categorical input features
twf = st.selectbox('Tool Wear Failure (TWF)', [0, 1])
hdf = st.selectbox('Heat Dissipation Failure (HDF)', [0, 1])
pwf = st.selectbox('Power Failure (PWF)', [0, 1])
osf = st.selectbox('Overstrain Failure (OSF)', [0, 1])

# Predict button
if st.button('Predict'):
    # Prepare input data
    input_data = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear, twf, hdf, pwf, osf]])
    input_data_pca = pca.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_pca)
    st.success(f'The predicted output is: {prediction[0]}')
