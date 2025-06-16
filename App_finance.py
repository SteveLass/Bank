import streamlit as st
import joblib
import numpy as np

# Charger le modèle et les encodeurs
model = joblib.load("inclusion_model.pkl")
le_dict = joblib.load("encoders.pkl")

st.title("🔍 Prédiction de l'inclusion financière")

# Saisie utilisateur
location = st.selectbox("Zone", le_dict['location_type'].classes_.tolist())
cellphone = st.selectbox("Accès téléphone", le_dict['cellphone_access'].classes_.tolist())
household_size = st.number_input("Taille du foyer", min_value=1)
age = st.slider("Âge du répondant", 16, 100, 30)
gender = st.selectbox("Sexe", le_dict['gender_of_respondent'].classes_.tolist())
relation = st.selectbox("Lien avec le chef de ménage", le_dict['relationship_with_head'].classes_.tolist())
marital = st.selectbox("Statut matrimonial", le_dict['marital_status'].classes_.tolist())
education = st.selectbox("Niveau d’éducation", le_dict['education_level'].classes_.tolist())
job = st.selectbox("Type d’emploi", le_dict['job_type'].classes_.tolist())

# Transformer en valeurs numériques
location = le_dict['location_type'].transform([location])[0]
cellphone = le_dict['cellphone_access'].transform([cellphone])[0]
gender = le_dict['gender_of_respondent'].transform([gender])[0]
relation = le_dict['relationship_with_head'].transform([relation])[0]
marital = le_dict['marital_status'].transform([marital])[0]
education = le_dict['education_level'].transform([education])[0]
job = le_dict['job_type'].transform([job])[0]

# Créer l’entrée pour la prédiction
features = np.array([[location, cellphone, household_size, age, gender, relation, marital, education, job]])

# Prédiction
if st.button("Prédire"):
    pred = model.predict(features)[0]
    st.success("✅ Cette personne est incluse financièrement." if pred == 1 else "❌ Cette personne n’a pas de compte bancaire.")
