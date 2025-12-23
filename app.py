import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Employee Salary Prediction",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #2c3e50;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #7f8c8d;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
.prediction {
    font-size: 22px;
    font-weight: bold;
    color: #27ae60;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="main-title">💼 Employee Salary Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Linear vs Polynomial Regression Visualization</div><br>', unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv(r'C:\Users\ADMIN\Downloads\emp_sal.csv')

dataset = load_data()

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# ---------------- MODELS ----------------
lin_reg = LinearRegression()
lin_reg.fit(x, y)

poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(x_poly, y)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")
model_type = st.sidebar.radio(
    "Select Regression Model",
    ("Linear Regression", "Polynomial Regression")
)

position_level = st.sidebar.slider(
    "Select Position Level",
    min_value=float(x.min()),
    max_value=float(x.max()),
    step=0.1,
    value=6.5
)

# ---------------- PREDICTION ----------------
if model_type == "Linear Regression":
    prediction = lin_reg.predict([[position_level]])[0]
else:
    prediction = lin_reg_poly.predict(poly_reg.transform([[position_level]]))[0]

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")
    st.markdown(
        f"<div class='prediction'>Predicted Salary: ₹ {prediction:,.2f}</div>",
        unsafe_allow_html=True
    )
    st.write("Model Used:", model_type)
    st.write("Position Level:", position_level)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Regression Visualization")

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='red', label='Actual Data')

    if model_type == "Linear Regression":
        plt.plot(x, lin_reg.predict(x), color='blue', label='Linear Regression')
    else:
        x_grid = np.linspace(min(x), max(x), 100).reshape(-1, 1)
        plt.plot(
            x_grid,
            lin_reg_poly.predict(poly_reg.transform(x_grid)),
            color='blue',
            label='Polynomial Regression (Degree 5)'
        )

    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.legend()
    st.pyplot(plt)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<br><center>🚀 Built with Streamlit & Scikit-learn</center>", unsafe_allow_html=True)
