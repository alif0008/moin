import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

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

# Train models
lr_country = LinearRegression().fit(X_train, y_country_train)
rf_country = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_country_train)

lr_code = LinearRegression().fit(X_train, y_code_train)
rf_code = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_code_train)

lr_year = LinearRegression().fit(X_train, y_year_train)
rf_year = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_year_train)

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
    
    # Linear Regression predictions
    lr_country_pred = le_country.inverse_transform([round(lr_country.predict(input_data)[0])])[0]
    lr_code_pred = le_code.inverse_transform([round(lr_code.predict(input_data)[0])])[0]
    lr_year_pred = round(lr_year.predict(input_data)[0])

    # Random Forest predictions
    rf_country_pred = le_country.inverse_transform([round(rf_country.predict(input_data)[0])])[0]
    rf_code_pred = le_code.inverse_transform([round(rf_code.predict(input_data)[0])])[0]
    rf_year_pred = round(rf_year.predict(input_data)[0])

    # Display results
    st.header("Predictions")
    st.subheader("Linear Regression")
    st.write(f"Country: {lr_country_pred}")
    st.write(f"Code: {lr_code_pred}")
    st.write(f"Year: {lr_year_pred}")

    st.subheader("Random Forest")
    st.write(f"Country: {rf_country_pred}")
    st.write(f"Code: {rf_code_pred}")
    st.write(f"Year: {rf_year_pred}")
