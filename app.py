import streamlit as st
import joblib
import numpy as np

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Prédiction du trafic routier en fonction de la météo", layout="centered")

# --- ÉTAPE 1 : Chargement des Assets (Modèle RF + Scalers) ---
@st.cache_resource
def load_assets():
    # Chemins à adapter selon ton environnement local
    # path = '/home/esama/ATUT 2025/Projet fil conducteur/'
    
    # model = joblib.load(path + 'rf_model_traffic.joblib')
    # scaler_x = joblib.load(path + 'rf_scaler_x.joblib')
    # scaler_y = joblib.load(path + 'rf_scaler_y.joblib')

    model = joblib.load('rf_model_traffic.joblib')
    scaler_x = joblib.load('rf_scaler_x.joblib')
    scaler_y = joblib.load('rf_scaler_y.joblib')
    
    return model, scaler_x, scaler_y

# Chargement effectif
model, scaler_x, scaler_y = load_assets()

# --- ÉTAPE 2 : Interface Utilisateur ---
st.title("🚗 Prédiction du Trafic Routier")
st.markdown("Saisissez les conditions actuelles pour obtenir une estimation du volume de trafic.")

with st.sidebar:
    st.header("⚙️ Paramètres d'entrée")
    
    # Variables temporelles
    heure = st.slider("Heure de la journée", 0, 23, 12)
    jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    jour_choisi = st.selectbox("Jour de la semaine", jours)
    
    st.divider()
    
    # Variables météo
    meteo_options = ['Clear', 'Clouds', 'Rain', 'Drizzle', 'Thunderstorm', 'Snow', 'Mist', 'Fog', 'Haze']
    meteo_choisie = st.selectbox("Conditions météo", meteo_options)
    
    temp = st.number_input("Température (Kelvin)", value=285.0)
    hum = st.number_input("Humidité (%)", value=50)
    wind = st.number_input("Vitesse du vent", value=5.0)
    visib = st.number_input("Visibilité (miles)", value=10)
    clouds = st.slider("Couverture nuageuse (%)", 0, 100, 20)
    rain = st.number_input("Pluie (mm/h)", value=0.0)
    snow = st.number_input("Neige (mm/h)", value=0.0)
    
    is_holiday = st.checkbox("Jour férié ?", value=False)

# --- ÉTAPE 3 : Backend & Transformation ---

# 1. Calcul des composantes cycliques pour l'heure
h_sin = np.sin(2 * np.pi * heure / 24)
h_cos = np.cos(2 * np.pi * heure / 24)

# 2. Mapping numérique
d_week = jours.index(jour_choisi)
weather_mapping = {m: i for i, m in enumerate(meteo_options)}
w_type = weather_mapping.get(meteo_choisie, 0)
holiday_val = 1 if is_holiday else 0

# --- ÉTAPE 4 : Prédiction ---

if st.button("📊 Prédire le volume de trafic"):
    # Construction du vecteur (Ordre identique à l'entraînement)
    # [is_holiday, humidity, wind_speed, visibility, temp, rain, snow, clouds, weather, h_sin, h_cos, d_week]
    raw_data = np.array([[
        holiday_val, hum, wind, visib, temp, rain, snow, clouds, w_type, h_sin, h_cos, d_week
    ]])
    
    # 1. Mise à l'échelle (Scaling)
    scaled_data = scaler_x.transform(raw_data)
    
    # 2. Prédiction avec Random Forest (Directement en 2D)
    pred_scaled = model.predict(scaled_data)
    
    # 3. Retour aux unités réelles (Inverse Scaling)
    res_reelle = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
    
    # --- AFFICHAGE DU RÉSULTAT ---
    resultat = int(res_reelle[0][0])
    st.balloons()
    st.metric(label="Volume de trafic estimé", value=f"{resultat} véhicules")
    
    # Petit message contextuel
    if resultat > 4000:
        st.warning("⚠️ Trafic dense prévu.")
    elif resultat < 1000:
        st.success("✅ Trafic fluide.")