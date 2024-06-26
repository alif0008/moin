import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Function to add background image from URL
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images3.alphacoders.com/128/thumb-1920-1287678.jpg");
             background-attachment: fixed;
             background-size: cover;
         }}
         .stApp::before {{
             content: "";
             position: absolute;
             top: 0;
             right: 0;
             bottom: 0;
             left: 0;
             background-color: rgba(0, 0, 0, 0.7);  /* Dark overlay */
         }}
         .stApp > div {{
             position: relative;
             z-index: 1;
         }}
         .stApp {{
             color: white;
         }}
         .stSelectbox label, .stNumberInput label {{
             color: white !important;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Load the dataset
@st.cache_data
def load_data():
    # Replace this with your actual dataset loading code
    df = pd.read_csv("prevalence-by-mental-and-substance-use-disorder.csv")
    return df

df = load_data()

# Prepare the data
X = df[['Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)',
        'Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)',
        'Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)',
        'Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)',
        'Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)',
        'Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)',
        'Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)']]

y_country = df['Country']
y_code = df['Code']
y_year = df['Year']

# Encode categorical variables
le_country = LabelEncoder()
le_code = LabelEncoder()
y_country_encoded = le_country.fit_transform(y_country)
y_code_encoded = le_code.fit_transform(y_code)

# Split the data
X_train, X_test, y_country_train, y_country_test, y_code_train, y_code_test, y_year_train, y_year_test = train_test_split(
    X, y_country_encoded, y_code_encoded, y_year, test_size=0.2, random_state=42)

# Train and evaluate models
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse

# Country models
lr_country, lr_country_mse = train_and_evaluate(LinearRegression(), X_train, X_test, y_country_train, y_country_test)
rf_country, rf_country_mse = train_and_evaluate(RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test, y_country_train, y_country_test)

# Code models
lr_code, lr_code_mse = train_and_evaluate(LinearRegression(), X_train, X_test, y_code_train, y_code_test)
rf_code, rf_code_mse = train_and_evaluate(RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test, y_code_train, y_code_test)

# Year models
lr_year, lr_year_mse = train_and_evaluate(LinearRegression(), X_train, X_test, y_year_train, y_year_test)
rf_year, rf_year_mse = train_and_evaluate(RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test, y_year_train, y_year_test)

# Choose the best models
country_model = rf_country if rf_country_mse < lr_country_mse else lr_country
code_model = rf_code if rf_code_mse < lr_code_mse else lr_code
year_model = rf_year if rf_year_mse < lr_year_mse else lr_year

# Add background image
add_bg_from_url()

# Streamlit app
st.title("Mental Health Prevalence Predictor")

# Input fields
st.header("Enter Mental Health Prevalence Data")
schizophrenia = st.number_input("Prevalence - Schizophrenia", min_value=0.1, max_value=1.0, step=0.1)
bipolar = st.number_input("Prevalence - Bipolar disorder", min_value=0.1, max_value=1.0, step=0.1)
eating_disorders = st.number_input("Prevalence - Eating disorders", min_value=0.1, max_value=1.0, step=0.1)
anxiety = st.number_input("Prevalence - Anxiety disorders", min_value=0.1, max_value=1.0, step=0.1)
drug_use = st.number_input("Prevalence - Drug use disorders", min_value=0.1, max_value=1.0, step=0.1)
depression = st.number_input("Prevalence - Depressive disorders", min_value=0.1, max_value=1.0, step=0.1)
alcohol_use = st.number_input("Prevalence - Alcohol use disorders", min_value=0.1, max_value=1.0, step=0.1)

# Make predictions
if st.button("Predict"):
    input_data = np.array([[schizophrenia, bipolar, eating_disorders, anxiety, drug_use, depression, alcohol_use]])
    
    country_pred = le_country.inverse_transform([round(country_model.predict(input_data)[0])])[0]
    code_pred = le_code.inverse_transform([round(code_model.predict(input_data)[0])])[0]
    year_pred = round(year_model.predict(input_data)[0])

    # Display results
    st.header("Predictions")
    st.write(f"Country: {country_pred}")
    st.write(f"Code: {code_pred}")
    st.write(f"Year: {year_pred}")
