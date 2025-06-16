import streamlit as st
import joblib
import numpy as np
import os

st.title("🔍 Prédiction de l'inclusion financière en Afrique")

# 🔒 Vérification des fichiers nécessaires
if not os.path.exists("inclusion_model.pkl"):
    st.error("❌ Le fichier 'inclusion_model.pkl1' est introuvable. Veuillez l’ajouter au dépôt GitHub.")
    st.stop()

if not os.path.exists("encoders.pkl"):
    st.error("❌ Le fichier 'encoders.pkl1' est introuvable. Veuillez l’ajouter au dépôt GitHub.")
    st.stop()

# ✅ Chargement du modèle et des encodeurs
model = joblib.load("inclusion_model.pkl1")
le_dict = joblib.load("encoders.pkl")

# 📝 Interface utilisateur
location = st.selectbox("Zone", le_dict['location_type'].classes_.tolist())
cellphone = st.selectbox("Accès téléphone", le_dict['cellphone_access'].classes_.tolist())
household_size = st.number_input("Taille du foyer", min_value=1)
age = st.slider("Âge du répondant", 16, 100, 30)
gender = st.selectbox("Sexe", le_dict['gender_of_respondent'].classes_.tolist())
relation = st.selectbox("Lien avec le chef de ménage", le_dict['relationship_with_head'].classes_.tolist())
marital = st.selectbox("Statut matrimonial", le_dict['marital_status'].classes_.tolist())
education = st.selectbox("Niveau d’éducation", le_dict['education_level'].classes_.tolist())
job = st.selectbox("Type d’emploi", le_dict['job_type'].classes_.tolist())

# 🔄 Encodage des variables
location = le_dict['location_type'].transform([location])[0]
cellphone = le_dict['cellphone_access'].transform([cellphone])[0]
gender = le_dict['gender_of_respondent'].transform([gender])[0]
relation = le_dict['relationship_with_head'].transform([relation])[0]
marital = le_dict['marital_status'].transform([marital])[0]
education = le_dict['education_level'].transform([education])[0]
job = le_dict['job_type'].transform([job])[0]

# 📊 Construction des features
features = np.array([[location, cellphone, household_size, age, gender, relation, marital, education, job]])

# 🔮 Prédiction
if st.button("Prédire"):
    pred = model.predict(features)[0]
    if pred == 1:
        st.success("✅ Cette personne est incluse financièrement (possède un compte bancaire).")
    else:
        st.warning("❌ Cette personne **n’est pas** incluse financièrement (ne possède pas de compte bancaire).")
