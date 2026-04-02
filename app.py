import streamlit as st

# Import หน้าเว็บแต่ละหน้า
import about_ml
import about_nn
import ml_demo
import nn_demo

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ML & NN Web App", layout="wide")

# สร้าง Sidebar สำหรับ Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("เลือกหน้า", ["📖 Machine Learning", "📖 Neural Network", "🧪 Demo ML", "🧪 Demo NN"])

# โหลดหน้าเว็บตามที่เลือก
if page == "📖 Machine Learning":
    about_ml.show()
elif page == "📖 Neural Network":
    about_nn.show()
elif page == "🧪 Demo ML":
    ml_demo.show()
elif page == "🧪 Demo NN":
    nn_demo.show()