# Smart Healthcare Assistant - Unified & Polished Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import html

# --- Configuration ---
warnings.filterwarnings("ignore")

st.set_page_config(page_title="🧠 Smart Healthcare Assistant", layout="wide")
st.title("🧠 Smart Healthcare Assistant")

st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
        font-weight:600;
    }
    .section-header {
        font-size:22px;
        font-weight:700;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #a2d5f2;
        padding-bottom: 0.3rem;
    }
    .precaution-list, .workout-list, .diet-list, .medication-list {
        padding-left: 1.2rem;
    }
    .drug-review {
        font-style: italic;
        background-color: #f0f4f8;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="big-font">
    This assistant predicts disease risks based on symptoms and health data.<br>
    Supports: <b>COVID-19</b>, <b>Heart Disease</b>, <b>Diabetes</b>, and <b>General Disease</b>.
    </p>
    """,
    unsafe_allow_html=True,
)

# Get base path of this script (works on any platform)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct relative paths
MODELS_DIR = os.path.join(BASE_DIR, "Models")
DATA_DIR = os.path.join(BASE_DIR, "data")
# Read the CSV file
drug_reviews_df = pd.read_csv(os.path.join(DATA_DIR, "Drug_Data_CSV.csv"))


# Helper function to show disease details (unchanged)
def helper(disease):
    disease_lower = disease.lower()
    desc_df = data_dfs['description_df']
    precautions_df = data_dfs['precautions_df']
    medications_df = data_dfs['medications_df']
    diets_df = data_dfs['diets_df']
    workout_df = data_dfs['workout_df']

    # Description
    desc_row = desc_df[desc_df["disease"].str.lower() == disease_lower]
    desc = desc_row["Description"].values[0] if not desc_row.empty else "No description available."

    # Precautions
    prec_row = precautions_df[precautions_df["disease"].str.lower() == disease_lower]
    precautions = prec_row.iloc[0, 1:].dropna().tolist() if not prec_row.empty else ["No precautions available."]

    # Medications
    med_row = medications_df[medications_df["disease"].str.lower() == disease_lower]
    medicines = med_row["Medication"].dropna().tolist() if not med_row.empty else ["No medications available."]

    # Drug Reviews (from the drug_reviews_df)
    drug_row = drug_reviews_df[drug_reviews_df["prescribed_for"].str.lower() == disease_lower]
    if not drug_row.empty:
        drugs = drug_row[["drugName", "Drug_Review"]].head(3).to_dict(orient='records')
    else:
        drugs = [{"drugName": "No drug found", "Drug_Review": "No review available."}]

    # Diet
    diet_row = diets_df[diets_df["disease"].str.lower() == disease_lower]
    diets = diet_row["Diet"].dropna().tolist() if not diet_row.empty else ["No diet information available."]

    # Workout
    workout_row = workout_df[workout_df["disease"].str.lower() == disease_lower]
    workouts = workout_row["workout"].dropna().tolist() if not workout_row.empty else [
        "No workout recommendations available."]

    return desc, precautions, medicines, drugs, diets, workouts


# --- Load Data & Models ---
@st.cache_resource
def load_models():
    return {
        "covid_model": joblib.load(f"{MODELS_DIR}/covid_rf_model.pkl"),
        "scaler": joblib.load(f"{MODELS_DIR}/scaler.pkl"),
        "disease_model": joblib.load(f"{MODELS_DIR}/RandomForestClassifier.pkl"),
        "label_encoder": joblib.load(f"{MODELS_DIR}/label_encoder.pkl"),
        "heart_model": joblib.load(f"{MODELS_DIR}/heart_disease_model.pkl"),
        "diabetes_model": joblib.load(f"{MODELS_DIR}/diabetes_logistic_model.pkl"),
        "diabetes_scaler": joblib.load(f"{MODELS_DIR}/diabetes_scaler.pkl")
    }


@st.cache_data
def load_data():
    dfs = {
        "description_df": pd.read_csv(f"{DATA_DIR}/description.csv"),
        "diets_df": pd.read_csv(f"{DATA_DIR}/diets.csv"),
        "medications_df": pd.read_csv(f"{DATA_DIR}/medications.csv"),
        "precautions_df": pd.read_csv(f"{DATA_DIR}/precautions_df.csv"),
        "workout_df": pd.read_csv(f"{DATA_DIR}/workout_df.csv"),
        "training_df": pd.read_csv(f"{DATA_DIR}/Training.csv"),
        "medicine_data": pd.read_csv(f"{DATA_DIR}/Drug_Data_CSV.csv")
    }
    dfs['medicine_data']['Drug_Review'] = dfs['medicine_data']['Drug_Review'].apply(html.unescape)
    return dfs


models = load_models()
data_dfs = load_data()

# --- Tabs UI Layout ---

# --- Navigation ---
st.sidebar.title("🧭 Navigation")
selected_tab = st.sidebar.radio("Go to",
                                ["🏠 Home", "🦠 COVID-19", "🩺 General Disease", "❤️ Heart Disease", "🧬 Diabetes"])

# --- HOME TAB ---
if selected_tab == "🏠 Home":
    st.header("Welcome to the Smart Healthcare Assistant")
    st.markdown("""
     <div style="font-size:18px; line-height:1.6;">
     This assistant helps in:
     <ul>
         <li>Predicting <b>COVID-19 risk</b></li>
         <li>Analyzing <b>Heart Disease</b></li>
         <li>Predicting <b>Diabetes</b></li>
         <li>Diagnosing <b>General Diseases</b> based on symptoms</li>
     </ul>
     <p>🔍 Use the sidebar to navigate through various health check modules.</p>
     </div>
     """, unsafe_allow_html=True)

# ---------------------------- COVID-19 TAB ----------------------------
if selected_tab == "🦠 COVID-19":
    st.header("🦠 COVID-19 Prediction")

    with st.form("covid_form"):
        age = st.slider("Age", 0, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        temp = st.slider("Body Temperature (°F)", 95.0, 105.0, 98.6, step=0.1)

        with st.expander("Select Symptoms and Conditions"):
            col1, col2, col3 = st.columns(3)
            with col1:
                dry_cough = st.checkbox("Dry Cough")
                weakness = st.checkbox("Weakness")
                drowsiness = st.checkbox("Drowsiness")
            with col2:
                sore_throat = st.checkbox("Sore Throat")
                breathing = st.checkbox("Breathing Problem")
                chest_pain = st.checkbox("Pain in Chest")
            with col3:
                appetite = st.checkbox("Change in Appetite")
                smell_loss = st.checkbox("Loss of Smell")

            col4, col5, col6 = st.columns(3)
            with col4:
                diabetes = st.checkbox("Diabetes")
                lung_disease = st.checkbox("Lung Disease")
            with col5:
                heart_disease = st.checkbox("Heart Disease")
                bp = st.checkbox("High BP")
            with col6:
                stroke = st.checkbox("Stroke")
                kidney = st.checkbox("Kidney Disease")

        submit = st.form_submit_button("Predict COVID Risk")

    if submit:
        features = np.array([[  # 17 values
            age, 1 if gender == "Male" else 0, temp,
            dry_cough, sore_throat, weakness,
            breathing, drowsiness, chest_pain,
            diabetes, heart_disease, lung_disease,
            stroke, bp, kidney,
            appetite, smell_loss
        ]], dtype=np.float64)
        
            
        
        # Assuming 'age' is at index 0 and 'temp' is at index 2
        scaled_values = models['scaler'].transform(features[:, [0, 2]])
        features[:, [0, 2]] = scaled_values

        # Get prediction and probability
        prediction = models['covid_model'].predict(features)[0]
        probability = models['covid_model'].predict_proba(features)[0][1]

        # Display result
        if prediction == 1:
            st.error(
                f"⚠️ You may be at risk for COVID-19.\n\n🧪 Probability: **{probability:.2%}**\nPlease consult a doctor."
            )
        
        elif 0.2 < probability < 0.5 or temp >= 99.8:
            if temp >= 99.8:
                st.warning("🌡️ Your body temperature seems above the normal range.\n🤒 Your symptoms may be related to an illness other than COVID-19.\nPlease consult a doctor.")
            else:
                st.warning("🤒 Your symptoms may be related to an illness other than COVID-19.\nPlease visit your nearest health center.")
        
        else:
            st.success("✅ Good news! Your symptoms don’t strongly indicate COVID-19.")


# ---------------------------- GENERAL DISEASE TAB ----------------------------
elif selected_tab == "🩺 General Disease":
    st.header("🩺 General Disease Prediction")

    training_df = data_dfs['training_df']
    label_encoder = models['label_encoder']
    disease_model = models['disease_model']
    disease_dict = {i: label for i, label in enumerate(label_encoder.classes_)}
    symptoms_dict = {sym.lower(): idx for idx, sym in enumerate(training_df.columns[:-1])}

    input_symptoms = st.text_input("Enter symptoms (comma-separated):")

    with st.expander("Show Available Symptoms"):
        st.markdown(", ".join(symptoms_dict.keys()))

    if st.button("Predict Disease"):
        if not input_symptoms:
            st.warning("Please enter at least one symptom.")
        else:
            input_vector = np.zeros(len(symptoms_dict))
            symptoms = [s.strip().lower().replace(" ", "_") for s in input_symptoms.split(",")]
            for s in symptoms:
                if s in symptoms_dict:
                    input_vector[symptoms_dict[s]] = 1
            prediction = disease_model.predict([input_vector])[0]
            predicted_disease = disease_dict[prediction]

            st.success(f"Predicted Disease: {predicted_disease}")

            # Additional Suggestions
            st.markdown('<div class="section-header">📝 Suggested Information</div>', unsafe_allow_html=True)

            desc, precautions, medicines, drugs, diets, workouts = helper(predicted_disease)

            st.markdown("### 🩺 Description")
            st.info(desc)

            st.markdown("### 🛡️ Precautions")
            st.markdown("<ol class='precaution-list'>", unsafe_allow_html=True)
            for p in precautions:
                st.markdown(f"<li>{p}</li>", unsafe_allow_html=True)
            st.markdown("</ol>", unsafe_allow_html=True)

            st.markdown("### 💊 Medications")
            st.markdown("<ul class='medication-list'>", unsafe_allow_html=True)
            for med in medicines:
                st.markdown(f"<li>{med}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)

            for idx, item in enumerate(drugs, 1):
                st.markdown(
                    f"<div class='drug-review'><b>{idx}. Drug Name:</b> {item['drugName']}<br><b>Review:</b> {item['Drug_Review']}</div>",
                    unsafe_allow_html=True)

            st.markdown("### 🥗 Diet Recommendations")
            st.markdown("<ul class='diet-list'>", unsafe_allow_html=True)
            for d in diets:
                st.markdown(f"<li>{d}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)

            st.markdown("### 🏋️ Workout Recommendations")
            st.markdown("<ol class='workout-list'>", unsafe_allow_html=True)
            for w in workouts:
                st.markdown(f"<li>{w}</li>", unsafe_allow_html=True)
            st.markdown("</ol>", unsafe_allow_html=True)

# ---------------------------- HEART DISEASE TAB ----------------------------
elif selected_tab == "❤️ Heart Disease":
    st.header("❤️ Heart Disease Prediction")

    with st.form("heart_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 18, 100, 50)
            sex = st.selectbox("Sex", ['M', 'F'])
            cp = st.selectbox("Chest Pain Type", ['TA', 'ATA', 'NAP', 'ASY'])
            rbp = st.number_input("Resting BP", 80, 200, 120)
            chol = st.number_input("Cholesterol", 100, 600, 200)
        with col2:
            fbs = st.selectbox("Fasting Sugar >120", [0, 1])
            ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
            max_hr = st.slider("Max HR", 60, 220, 150)
            angina = st.selectbox("Exercise Angina", ['Y', 'N'])
            oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
            slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

        submit = st.form_submit_button("Predict Heart Disease")

    if submit:
        df = pd.DataFrame([{"Age": age, "Sex": sex, "ChestPainType": cp, "RestingBP": rbp,
                            "Cholesterol": chol, "FastingBS": fbs, "RestingECG": ecg,
                            "MaxHR": max_hr, "ExerciseAngina": angina, "Oldpeak": oldpeak,
                            "ST_Slope": slope}])
        pred = models['heart_model'].predict(df)[0]
        proba = models['heart_model'].predict_proba(df)[0][1]

        if pred == 1:
            st.error(f"Likely to have heart disease (probability: {proba:.2f})")
            desc, precautions, medicines, drugs, diets, workouts = helper('heart attack')

            st.markdown("### 🩺 Description")
            st.info(desc)

            st.markdown("### 🛡️ Precautions")
            st.markdown("<ol class='precaution-list'>", unsafe_allow_html=True)
            for p in precautions:
                st.markdown(f"<li>{p}</li>", unsafe_allow_html=True)
            st.markdown("</ol>", unsafe_allow_html=True)

            st.markdown("### 💊 Medications")
            st.markdown("<ul class='medication-list'>", unsafe_allow_html=True)
            for med in medicines:
                st.markdown(f"<li>{med}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)

            for idx, item in enumerate(drugs, 1):
                st.markdown(
                    f"<div class='drug-review'><b>{idx}. Drug Name:</b> {item['drugName']}<br><b>Review:</b> {item['Drug_Review']}</div>",
                    unsafe_allow_html=True)

            st.markdown("### 🥗 Diet Recommendations")
            st.markdown("<ul class='diet-list'>", unsafe_allow_html=True)
            for d in diets:
                st.markdown(f"<li>{d}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)

            st.markdown("### 🏋️ Workout Recommendations")
            st.markdown("<ol class='workout-list'>", unsafe_allow_html=True)
            for w in workouts:
                st.markdown(f"<li>{w}</li>", unsafe_allow_html=True)
            st.markdown("</ol>", unsafe_allow_html=True)
        else:
            st.success(f"Unlikely to have heart disease (probability: {proba:.2f})")

# ---------------------------- DIABETES TAB ----------------------------
elif selected_tab == "🧬 Diabetes":
    st.header("🧬 Diabetes Prediction")

    col1, col2 = st.columns(2)
    with col1:
        preg = st.number_input("Pregnancies", 0, 20, 2)
        gluc = st.number_input("Glucose", 0, 300, 120)
        bp = st.number_input("Blood Pressure", 0, 200, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)
    with col2:
        ins = st.number_input("Insulin", 0, 900, 79)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.6)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.3)
        age = st.number_input("Age", 0, 120, 32)

    if st.button("Predict Diabetes"):
        df = pd.DataFrame([{
            'Pregnancies': preg, 'Glucose': gluc, 'BloodPressure': bp,
            'SkinThickness': skin, 'Insulin': ins, 'BMI': bmi,
            'DiabetesPedigreeFunction': dpf, 'Age': age
        }])
        scaled = models['diabetes_scaler'].transform(df)
        pred = models['diabetes_model'].predict(scaled)[0]
        prob = models['diabetes_model'].predict_proba(scaled)[0][1]

        if pred == 1 or prob > 0.5:
            st.error(f"Likely to have Diabetes (probability: {prob:.2f})")
            desc, precautions, medicines, drugs, diets, workouts = helper('diabetes')

            st.markdown("### 🩺 Description")
            st.info(desc)

            st.markdown("### 🛡️ Precautions")
            st.markdown("<ol class='precaution-list'>", unsafe_allow_html=True)
            for p in precautions:
                st.markdown(f"<li>{p}</li>", unsafe_allow_html=True)
            st.markdown("</ol>", unsafe_allow_html=True)

            st.markdown("### 💊 Medications")
            st.markdown("<ul class='medication-list'>", unsafe_allow_html=True)
            for med in medicines:
                st.markdown(f"<li>{med}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)

            for idx, item in enumerate(drugs, 1):
                st.markdown(
                    f"<div class='drug-review'><b>{idx}. Drug Name:</b> {item['drugName']}<br><b>Review:</b> {item['Drug_Review']}</div>",
                    unsafe_allow_html=True)

            st.markdown("### 🥗 Diet Recommendations")
            st.markdown("<ul class='diet-list'>", unsafe_allow_html=True)
            for d in diets:
                st.markdown(f"<li>{d}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)

            st.markdown("### 🏋️ Workout Recommendations")
            st.markdown("<ol class='workout-list'>", unsafe_allow_html=True)
            for w in workouts:
                st.markdown(f"<li>{w}</li>", unsafe_allow_html=True)
            st.markdown("</ol>", unsafe_allow_html=True)

        else:
            st.success(f"Unlikely to have Diabetes (probability: {prob:.2f})")
