import streamlit as st
import numpy as np
import joblib

# Load the trained models
lr = joblib.load('linear_regression_model.pkl')
rf = joblib.load('random_forest_model.pkl')

# Streamlit app
st.title('Mental Fitness Tracker')

# Input fields for user data
st.header('Input Your Data')
schizophrenia = st.number_input('Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)', min_value=0.0, max_value=100.0, value=0.0)
bipolar = st.number_input('Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)', min_value=0.0, max_value=100.0, value=0.0)
eating_disorders = st.number_input('Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)', min_value=0.0, max_value=100.0, value=0.0)
anxiety = st.number_input('Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)', min_value=0.0, max_value=100.0, value=0.0)
drug_use = st.number_input('Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)', min_value=0.0, max_value=100.0, value=0.0)
depressive = st.number_input('Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)', min_value=0.0, max_value=100.0, value=0.0)
alcohol_use = st.number_input('Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)', min_value=0.0, max_value=100.0, value=0.0)

# Make predictions
if st.button('Predict'):
    input_data = np.array([[schizophrenia, bipolar, eating_disorders, anxiety, drug_use, depressive, alcohol_use]])
    
    # Linear Regression Prediction
    lr_pred = lr.predict(input_data)
    
    # Random Forest Prediction
    rf_pred = rf.predict(input_data)
    
    st.subheader('Predictions')
    st.write("Linear Regression Prediction:")
    st.write(f"Country: {lr_pred[0][0]}")
    st.write(f"Code: {lr_pred[0][1]}")
    st.write(f"Year: {lr_pred[0][2]}")
    
    st.write("Random Forest Prediction:")
    st.write(f"Country: {rf_pred[0][0]}")
    st.write(f"Code: {rf_pred[0][1]}")
    st.write(f"Year: {rf_pred[0][2]}")

# Display information about the models
st.header('Model Information')
st.markdown("""
### Linear Regression Model
Linear Regression is a simple regression model that assumes a linear relationship between the input variables (features) and the output variable (target).

### Random Forest Regressor Model
Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the average prediction of the individual trees.
""")
