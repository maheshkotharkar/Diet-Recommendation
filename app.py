import streamlit as st
import joblib
import numpy as np
import random

st.markdown(
    """
    <style>
    /* Vibrant background image */
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1601313104473-4e18f5e3c1a9');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Center the title */
    .main > div:first-child {
        text-align: center;
        padding-top: 20px;
    }

    /* Customize title color */
    .stApp h1 {
        color: #2E8B57; /* Sea green */
    }

    /* Style buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }

    /* Add shadow to input widgets */
    .stTextInput, .stSelectbox, .stNumberInput {
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.85);  /* Slight background for contrast */
        border-radius: 10px;
    }

    /* Container spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load the trained model
model = joblib.load(r"C:\Users\madhu\OneDrive\Desktop\project\diet_plan_model_Randomf.pkl")
label_encoder = joblib.load(r"C:\Users\madhu\OneDrive\Desktop\project\label_encoder.pkl")

st.title("🥗 AI-Powered Diet Plan Recommendation System")
st.markdown(
    """
    <div style='text-align:center; padding: 10px; background-color:#f0f9ff; border-radius:10px; margin-bottom:20px'>
        <h3>💡 Personalized Health + Diet Plan Generator</h3>
        <p style='font-size:16px;'>Smart nutrition suggestions powered by Machine Learning </p>
    </div>
    """,
    unsafe_allow_html=True
)


# User Input Fields
#  Height & Weight inputs for BMI
height = st.number_input("Enter your height (in cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Enter your weight (in kg)", min_value=30, max_value=200, value=65)

#  Calculate BMI
bmi = round(weight / ((height / 100) ** 2), 2)
st.info(f"🧬 Your BMI is: **{bmi}**")

#  Automatically determine Health Goal
if bmi > 30:
    health_goal = "Weight Loss"
elif bmi < 18.5:
    health_goal = "Weight Gain"
else:
    health_goal = "Maintain Weight"

st.success(f"🎯 Based on your BMI, your health goal is: **{health_goal}**")

medical_conditions = st.selectbox("Medical Condition", ["None", "Diabetes", "Hypertension", "Heart Disease"])
macro_preference = st.selectbox("Macro Preference", ["Balanced", "High Protein", "Low Carb"])
diet_type_1 = st.selectbox("Diet Type 1", ["Vegetarian", "Non-Vegetarian", "Vegan"])
diet_type_2 = st.selectbox("Diet Type 2", ["Gluten-Free", "Regular"])

# Encode categorical values manually
medical_conditions_mapping = {"None": 0, "Diabetes": 1, "Hypertension": 2, "Heart Disease": 3}
macro_preference_mapping = {"Balanced": 0, "High Protein": 1, "Low Carb": 2}
diet_type_1_mapping = {"Vegetarian": 0, "Non-Vegetarian": 1, "Vegan": 2}
diet_type_2_mapping = {"Gluten-Free": 0, "Regular": 1}

# Convert selections to numerical values
medical_conditions_encoded = medical_conditions_mapping[medical_conditions]
macro_preference_encoded = macro_preference_mapping[macro_preference]
diet_type_1_encoded = diet_type_1_mapping[diet_type_1]
diet_type_2_encoded = diet_type_2_mapping[diet_type_2]

# Create input array
input_data = np.array([[medical_conditions_encoded, macro_preference_encoded, diet_type_1_encoded, diet_type_2_encoded]])

# Prediction button
if st.button("Get Diet Plan"):
    try:
        # Get top 3 meal plans based on probability
        probs = model.predict_proba(input_data)[0]
        top_3_indices = np.argsort(probs)[-3:][::-1]  # Get indices of top 3 meals
        top_3_meals = label_encoder.inverse_transform(top_3_indices)

        # Meal filters
        vegetarian_meals = [meal for meal in label_encoder.classes_ if "Chicken" not in meal and "Fish" not in meal and "Eggs" not in meal]
        vegan_meals = [meal for meal in vegetarian_meals if "Cheese" not in meal and "Milk" not in meal]
        non_vegetarian_meals = [meal for meal in label_encoder.classes_ if meal not in vegetarian_meals]

        # Adjust for diet types
        final_meals = []
        for meal in top_3_meals:
            if diet_type_1 == "Vegetarian" and meal in non_vegetarian_meals:
                meal = random.choice(vegetarian_meals) if vegetarian_meals else "Vegetarian Meal Plan (Default)"
            if diet_type_1 == "Vegan" and meal not in vegan_meals:
                meal = random.choice(vegan_meals) if vegan_meals else "Vegan Meal Plan (Default)"
            final_meals.append(meal)
        
        # Display top 3 meal recommendations
        st.success("🍽 Recommended Meal Plans:")
        for i, meal in enumerate(final_meals):
            st.write(f"{i+1}. **{meal}**")

    except ValueError:
        st.error("⚠️ Error: Predicted label is unknown. Please retrain the label encoder.")
