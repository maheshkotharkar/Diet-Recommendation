# app.py
import streamlit as st
import joblib
import numpy as np
import random
from pathlib import Path

# Initialize session state for meal history
if 'meal_history' not in st.session_state:
    st.session_state.meal_history = []

# ========== CSS Styling ==========
st.markdown("""
<style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1601313104473-4e18f5e3c1a9');
        background-size: cover;
        background-attachment: fixed;
    }
    .main > div:first-child {
        text-align: center;
        padding-top: 20px;
    }
    .stApp h1 {
        color: #2E8B57;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #3e8e41;
        transform: scale(1.01);
    }
    .input-container {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .meal-card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .meal-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ========== Model Loading ==========
@st.cache_resource
def load_model_assets():
    try:
        model_path = Path(__file__).parent / "diet_plan_model.pkl"
        encoder_path = Path(__file__).parent / "label_encoder.pkl"
        return joblib.load(model_path), joblib.load(encoder_path)
    except Exception as e:
        st.error(f"❌ Error loading model files: {str(e)}")
        st.error("Please ensure:")
        st.error("1. Model files exist in the same directory")
        st.error("2. Files are not corrupted")
        st.error("3. Dependencies match requirements.txt")
        st.stop()

model, label_encoder = load_model_assets()

# ========== App Header ==========
st.title("🥗 AI-Powered Diet Plan Recommendation")
st.markdown("""
<div style='text-align:center; background-color:rgba(240,249,255,0.8); border-radius:10px; padding:10px; margin-bottom:20px'>
    <h3>💡 Personalized Nutrition Plans</h3>
    <p style='font-size:16px;'>Get customized meal recommendations based on your profile</p>
</div>
""", unsafe_allow_html=True)

# ========== User Input Section ==========
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        height = st.number_input("Your height (cm)", min_value=100, max_value=250, value=170)
    with col2:
        weight = st.number_input("Your weight (kg)", min_value=30, max_value=200, value=65)
    
    # BMI Calculation
    bmi = round(weight / ((height / 100) ** 2), 2)
    bmi_status = (
        "Underweight" if bmi < 18.5 else
        "Normal" if 18.5 <= bmi < 25 else
        "Overweight" if 25 <= bmi < 30 else
        "Obese"
    )
    
    st.info(f"""
    🧬 **Health Summary**  
    BMI: {bmi} ({bmi_status})  
    Goal: {'Weight Gain' if bmi < 18.5 else 'Weight Loss' if bmi > 25 else 'Maintenance'}
    """)
    
    # Diet Preferences
    medical_conditions = st.selectbox(
        "Medical Considerations",
        ["None", "Diabetes", "Hypertension", "Heart Disease"]
    )
    
    col3, col4 = st.columns(2)
    with col3:
        diet_type = st.selectbox(
            "Diet Preference",
            ["Vegetarian", "Non-Vegetarian", "Vegan"]
        )
    with col4:
        macro_preference = st.selectbox(
            "Macronutrient Focus",
            ["Balanced", "High Protein", "Low Carb"]
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Prediction Logic ==========
if st.button("🍎 Generate Meal Plan", type="primary", use_container_width=True):
    with st.spinner('Analyzing your profile and generating recommendations...'):
        try:
            # Encoding mappings
            condition_map = {"None":0, "Diabetes":1, "Hypertension":2, "Heart Disease":3}
            macro_map = {"Balanced":0, "High Protein":1, "Low Carb":2}
            diet_map = {"Vegetarian":0, "Non-Vegetarian":1, "Vegan":2}
            
            # Prepare input data
            input_data = np.array([[
                condition_map[medical_conditions],
                macro_map[macro_preference],
                diet_map[diet_type]
            ]])
            
            # Get predictions
            probs = model.predict_proba(input_data)[0]
            top_3_indices = np.argsort(probs)[-3:][::-1]
            top_3_meals = label_encoder.inverse_transform(top_3_indices)
            
            # Filter meals based on diet type
            veg_meals = [m for m in label_encoder.classes_ 
                        if not any(nv in m for nv in ["Chicken", "Fish", "Eggs"])]
            vegan_meals = [m for m in veg_meals 
                         if not any(d in m for d in ["Cheese", "Milk", "Yogurt"])]
            
            # Adjust recommendations
            final_meals = []
            for meal in top_3_meals:
                if diet_type == "Vegetarian" and meal not in veg_meals:
                    meal = random.choice(veg_meals) if veg_meals else "Vegetarian Plate"
                elif diet_type == "Vegan" and meal not in vegan_meals:
                    meal = random.choice(vegan_meals) if vegan_meals else "Vegan Bowl"
                final_meals.append(meal)
            
            # Store in session history
            st.session_state.meal_history = final_meals
            
            # Display results
            st.success("🎉 Here are your personalized meal recommendations:")
            for i, meal in enumerate(final_meals, 1):
                st.markdown(f"""
                <div class="meal-card">
                    <h4>🥗 Option {i}: {meal}</h4>
                    <p>Macro Focus: {macro_preference} | Diet Type: {diet_type}</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"⚠️ Error generating recommendations: {str(e)}")

# ========== Meal History ==========
if st.session_state.meal_history:
    with st.expander("📚 Previous Recommendations"):
        for meal in st.session_state.meal_history:
            st.write(f"🍽️ {meal}")

# ========== Footer ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: small; color: gray;'>
    <p>Note: These recommendations are AI-generated and should not replace professional medical advice</p>
    <p>Model Version: 1.0 | Last Updated: 2023-11-15</p>
</div>
""", unsafe_allow_html=True)
