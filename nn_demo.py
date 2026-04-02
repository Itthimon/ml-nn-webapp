import streamlit as st
import joblib
import numpy as np

model = joblib.load("models/ramen_model.pkl")
scaler = joblib.load("models/ramen_scaler.pkl")
label_encoders = joblib.load("models/ramen_label_encoders.pkl")

def show():
    st.title("🧪 Demo: Ramen Neural Network Model")
    st.write("🔍 เลือกข้อมูลราเมนเพื่อทำนายคะแนน (`Stars`)")

    brand   = st.selectbox("🏭 Brand",   label_encoders["Brand"].classes_)
    variety = st.selectbox("🍜 Variety", label_encoders["Variety"].classes_)
    style   = st.selectbox("🥢 Style",   label_encoders["Style"].classes_)
    country = st.selectbox("🌍 Country", label_encoders["Country"].classes_)

    if st.button("🔮 ทำนายคะแนนราเมน"):
        input_data = np.array([
            label_encoders["Brand"].transform([brand])[0],
            label_encoders["Variety"].transform([variety])[0],
            label_encoders["Style"].transform([style])[0],
            label_encoders["Country"].transform([country])[0]
        ]).reshape(1, -1)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prediction = float(np.clip(prediction, 0, 5))
        st.success(f"⭐ โมเดลทำนายว่าราเมนนี้ได้คะแนน: **{prediction:.2f} Stars**")
