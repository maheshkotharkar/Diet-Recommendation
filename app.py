import streamlit as st
import joblib
import numpy as np
import random
from pathlib import Path

# Initialize session state
if 'meal_history' not in st.session_state:
    st.session_state.meal_history = []

# ====== Simplified CSS ======
st.markdown("""
<style>
.stApp {
    background-image: url('https://images.unsplash.com/photo-1601313104473-4e18f5e3c1a9');
    background-size: cover;
    background-attachment: fixed;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
}
.meal-card {
    background-color: rgba(255,255,255,0.95);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ====== Model Loading ======
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('diet_plan_model_Randomf.pkl')
        encoder = joblib.load('label_encoder.pkl')
        return model, encoder
    except Exception as e:
        st.error(f"Failed to load model files: {str(e)}")
        st.stop()

model, label_encoder = load_assets()

# ====== App Interface ======
st.title("🥗 AI-Powered Diet Plan Recommendation")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    height = st.number_input("Height (cm)", 100, 250, 170)
with col2:
    weight = st.number_input("Weight (kg)", 30, 200, 65)

bmi = round(weight / ((height/100)**2), 1)
st.info(f"BMI: {bmi} ({'Underweight' if bmi < 18.5 else 'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese'})")

diet_type = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])
health_goal = st.selectbox("Health Goal", ["Weight Loss", "Maintenance", "Weight Gain"])

# Prediction
if st.button("Get Recommendations"):
    try:
        # Simplified prediction logic
        meals = label_encoder.classes_
        if diet_type == "Vegetarian":
            meals = [m for m in meals if "Chicken" not in m and "Fish" not in m]
        elif diet_type == "Vegan":
            meals = [m for m in meals if "Chicken" not in m and "Fish" not in m and "Cheese" not in m]
        
        recommendations = random.sample(list(meals), min(3, len(meals)))
        st.session_state.meal_history = recommendations
        
        st.success("Recommended Meal Plans:")
        for i, meal in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="meal-card">
                <b>Option {i}:</b> {meal}
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")

# Footer
st.markdown("---")
st.caption("Note: These recommendations are AI-generated and should not replace professional medical advice")
