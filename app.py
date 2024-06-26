import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set the title of the Streamlit app
st.title('Mental Fitness Tracker')

# About the project
st.markdown("""
## Mental Fitness Tracker
The Mental Fitness Tracker is an AI-powered project aimed at monitoring and supporting mental well-being using advanced algorithms. 
It uses regression models to analyze and predict mental fitness based on various factors.
""")

# Function to train models (this would typically be done offline and models would be loaded directly)
def train_models():
    # Sample data (replace with your actual data)
    data = pd.DataFrame({
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100),
        'Target': np.random.rand(100)
    })
    
    # Define features and target variable
    x = data[['Feature1', 'Feature2']]
    y = data['Target']
    
    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train Linear Regression Model
    lr = LinearRegression()
    lr.fit(xtrain, ytrain)

    # Train Random Forest Regressor Model
    rf = RandomForestRegressor()
    rf.fit(xtrain, ytrain)
    
    return lr, rf

# Train models (or load pre-trained models)
lr, rf = train_models()

# Form for user input
st.header('Input Your Data')
feature1 = st.number_input('Feature 1', min_value=0.0, max_value=1.0, value=0.5)
feature2 = st.number_input('Feature 2', min_value=0.0, max_value=1.0, value=0.5)

# Make predictions
if st.button('Predict'):
    input_data = np.array([[feature1, feature2]])
    
    # Linear Regression Prediction
    lr_pred = lr.predict(input_data)
    
    # Random Forest Prediction
    rf_pred = rf.predict(input_data)
    
    st.subheader('Predictions')
    st.write(f"Linear Regression Prediction: {lr_pred[0]:.2f}")
    st.write(f"Random Forest Prediction: {rf_pred[0]:.2f}")

# Display information about the models
st.header('Model Information')
st.markdown("""
### Linear Regression Model
Linear Regression is a simple regression model that assumes a linear relationship between the input variables (features) and the output variable (target).

### Random Forest Regressor Model
Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the average prediction of the individual trees.
""")

