import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the pre-cleaned dataset
@st.cache_data
def load_data():
    return pd.read_csv('car_data.csv')

# Preprocess data and train the Random Forest model
@st.cache_resource
def train_model(data):
    # One-hot encode categorical features
    data_encoded = pd.get_dummies(data, drop_first=True)
    
    # Preprocessing
    X = data_encoded.drop(columns=['Price'])
    y = data_encoded['Price']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model, scaler, X_test, y_test, X.columns

# Load data and train model
data = load_data()
model, scaler, X_test, y_test, feature_columns = train_model(data)

# App layout
st.title("Car Recommendation System")
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select an option:", ["Home", "Predict Price", "Recommend Cars", "Visualizations", "Model Performance"])

if options == "Home":
    st.write("Welcome to the Car Recommendation System!")
    st.dataframe(data.head())

elif options == "Predict Price":
    st.header("Predict Car Price")
    
    # Add input widgets for features
    condition = st.selectbox("Condition", data['Condition'].unique())
    drivetrain = st.selectbox("Drivetrain", data['Drivetrain'].unique())
    fuel_type = st.selectbox("Fuel Type", data['Fuel type'].unique())
    max_mpg = st.slider("Max MPG", int(data['Max MPG'].min()), int(data['Max MPG'].max()), 30)
    year = st.slider("Year", int(data['Year'].min()), int(data['Year'].max()), 2020)
    mileage = st.slider("Mileage", int(data['Mileage'].min()), int(data['Mileage'].max()), 50000)
    
    # Prepare input data for prediction
    input_data = pd.DataFrame([{
        'Condition': condition, 
        'Drivetrain': drivetrain, 
        'Fuel type': fuel_type, 
        'Max MPG': max_mpg, 
        'Year': year, 
        'Mileage': mileage
    }])
    
    # One-hot encode and align with training data
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)
    input_data_aligned = input_data_encoded.reindex(columns=feature_columns, fill_value=0)
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data_aligned)
    
    # Predict price using Random Forest
    predicted_price = model.predict(input_data_scaled)
    st.write(f"Predicted Price: ${predicted_price[0]:,.2f}")

elif options == "Recommend Cars":
    st.header("Find Your Ideal Car")
    price_range = st.slider("Price Range", int(data['Price'].min()), int(data['Price'].max()), (5000, 20000))
    condition = st.multiselect("Condition", data['Condition'].unique())
    drivetrain = st.multiselect("Drivetrain", data['Drivetrain'].unique())
    fuel_type = st.multiselect("Fuel Type", data['Fuel type'].unique())
    
    recommendations = data[
        (data['Price'].between(*price_range)) & 
        (data['Condition'].isin(condition)) & 
        (data['Drivetrain'].isin(drivetrain)) & 
        (data['Fuel type'].isin(fuel_type))
    ]
    st.write("Recommended Cars:")
    st.dataframe(recommendations)

elif options == "Visualizations":
    st.header("Data Insights")
    # Price distribution
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data['Price'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
    
    # Scatter plot
    st.subheader("Feature Comparison")
    x_axis = st.selectbox("X-axis", data.select_dtypes(include=np.number).columns)
    y_axis = st.selectbox("Y-axis", data.select_dtypes(include=np.number).columns, index=1)
    fig, ax = plt.subplots()
    sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
    st.pyplot(fig)

elif options == "Model Performance":
    st.header("Model Performance Metrics")
    # Residual plot
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    st.subheader("Residuals Distribution")
    fig, ax = plt.subplots()
    sns.histplot(residuals, bins=30, kde=True, ax=ax)
    ax.set_title("Residuals Distribution")
    st.pyplot(fig)
    
    # R² score
    r2 = model.score(X_test, y_test)
    st.subheader("R² Score")
    st.write(f"The R² score of the Linear Regression model is: {r2:.2f}")

