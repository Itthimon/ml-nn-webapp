import streamlit as st
import joblib  # ✅ ต้อง import joblib
import numpy as np

# โหลดโมเดล, LabelEncoder และ Scaler
model = joblib.load("models/pokedex_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")  # ✅ โหลด StandardScaler

def show():
    st.title("🧪 Demo: Pokedex Machine Learning Model")
    
    attack = st.number_input("🛡️ Attack", min_value=0, max_value=200, value=50)
    defense = st.number_input("🛡️ Defense", min_value=0, max_value=200, value=50)
    hp = st.number_input("❤️ HP", min_value=0, max_value=255, value=60)
    speed = st.number_input("⚡ Speed", min_value=0, max_value=200, value=60)
    s_attack = st.number_input("🔥 Special Attack", min_value=0, max_value=200, value=50)
    s_defense = st.number_input("🛡️ Special Defense", min_value=0, max_value=200, value=50)

    if st.button("🔮 ทำนายประเภทโปเกมอน"):
        input_data = np.array([[attack, defense, hp, speed, s_attack, s_defense]])
        input_data_scaled = scaler.transform(input_data)  # ✅ ใช้ StandardScaler
        
        prediction_num = model.predict(input_data_scaled)[0]
        prediction_label = label_encoder.inverse_transform([prediction_num])[0]

        st.success(f"🎉 โมเดลทำนายว่าโปเกมอนนี้เป็นประเภท: **{prediction_label}**")
