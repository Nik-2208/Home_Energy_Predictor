import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
rf = joblib.load("rfr_model.pkl")

st.set_page_config(page_title="Smart Home Energy Predictor", page_icon="⚡", layout="wide")

# -----------------------------
# UI Header
# -----------------------------
st.markdown(
    """
    <div style="background-color:#4B9CD3;padding:15px;border-radius:10px">
        <h1 style="color:white;text-align:center;">Smart Home Energy Consumption Predictor ⚡</h1>
        <p style="color:white;text-align:center;">Predict your household's energy consumption based on appliances, weather, and time features</p>
    </div>
    """, unsafe_allow_html=True
)

st.write("## Enter the details to get prediction:")

# -----------------------------
# Input UI
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    appliance = st.selectbox("Appliance Type", [
        'Fridge', 'Oven', 'Microwave', 'Heater', 'Lights', 'TV', 'Dishwasher', 'Computer', 'Washing Machine'
    ])
    temperature = st.number_input("Outdoor Temperature (°C)", value=25.0, step=0.1)
    
with col2:
    season = st.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Fall'])
    household_size = st.number_input("Household Size", min_value=1, max_value=10, value=3, step=1)
    
with col3:
    date_input = st.date_input("Date", value=datetime.today())
    time_input = st.time_input("Time", value=datetime.now().time())

# -----------------------------
# Preprocess input for model
# -----------------------------
input_df = pd.DataFrame({
    'Appliance Type': [appliance],
    'Outdoor Temperature (°C)': [temperature],
    'Season': [season],
    'Household Size': [household_size],
    'Month': [date_input.month],
    'DayOfWeek': [date_input.weekday()],
    'Hour': [time_input.hour]
})

# One-hot encode categorical columns
X_cols = rf.feature_names_in_
input_df = pd.get_dummies(input_df, columns=['Appliance Type','Season'], drop_first=True)

# Ensure all columns exist
for col in X_cols:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[X_cols]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Energy Consumption ⚡"):
    prediction = rf.predict(input_df)[0]
    st.success(f"Predicted Energy Consumption: **{prediction:.2f} kWh**")
    
    # Layman-friendly explanation
    daily_equivalent = prediction * 30  # Assuming prediction is per use or per hour
    st.info(
        f"This means that your **household may consume approximately {prediction:.2f} kWh at this time**.\n\n"
        f"Over a month (~30 days) this is roughly **{daily_equivalent:.2f} kWh**, "
        "which can help estimate your electricity bill and understand appliance impact on energy usage."
    )

# -----------------------------
# Feature Importance Plot
# -----------------------------
st.write("## Feature Importance")
importances = pd.Series(rf.feature_importances_, index=X_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax)
ax.set_title("Feature Importance")
st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div style="background-color:#4B9CD3;padding:10px;border-radius:5px;margin-top:20px;">
        <p style="color:white;text-align:center;">Smart Home Energy Predictor ⚡ | Made with ❤️ by Nikhilesh Chavda (Github: Nik-2208)</p>
    </div>
    """, unsafe_allow_html=True
)
