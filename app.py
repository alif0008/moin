import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
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

# Upload data files
st.header('Upload Your Data Files')
uploaded_file1 = st.file_uploader("Upload the prevalence data CSV", type="csv")
uploaded_file2 = st.file_uploader("Upload the mental and substance use data CSV", type="csv")

if uploaded_file1 and uploaded_file2:
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)

    st.subheader('First Few Rows of Prevalence Data')
    st.write(df1.head())

    st.subheader('Last Few Rows of Mental and Substance Use Data')
    st.write(df2.tail(10))

    # Data Merging
    st.subheader('Merged Data')
    data = pd.merge(df1, df2, on="Year")
    st.write(data.head())

    # Display correlation matrix
    st.subheader('Correlation Matrix')
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot()

    # Define features and target variable
    x = data[['Some_Feature1', 'Some_Feature2']]  # Update with actual feature columns
    y = data['Target_Variable']  # Update with actual target variable

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    # Linear Regression Model
    st.subheader('Linear Regression Model')
    lr = LinearRegression()
    lr.fit(xtrain, ytrain)
    ytrain_pred = lr.predict(xtrain)
    mse = mean_squared_error(ytrain, ytrain_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(ytrain, ytrain_pred)
    st.write(f"MSE: {mse}")
    st.write(f"RMSE: {rmse}")
    st.write(f"R2 Score: {r2}")

    # Random Forest Regressor Model
    st.subheader('Random Forest Regressor Model')
    rf = RandomForestRegressor()
    rf.fit(xtrain, ytrain)
    ytrain_pred = rf.predict(xtrain)
    mse = mean_squared_error(ytrain, ytrain_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(ytrain, ytrain_pred)
    st.write(f"MSE: {mse}")
    st.write(f"RMSE: {rmse}")
    st.write(f"R2 Score: {r2}")

    # Model Comparison
    st.subheader('Model Comparison on Test Set')
    lr_scores = lr.score(xtest, ytest)
    rf_scores = rf.score(xtest, ytest)
    st.write(f"Linear Regression Test Score: {lr_scores}")
    st.write(f"Random Forest Regressor Test Score: {rf_scores}")

else:
    st.write("Please upload both CSV files to proceed.")

