import streamlit as st

def show():
    st.title("📖 การพัฒนาโมเดล Neural Network (Ramen Ratings)")

    st.header("📌 แหล่งที่มาของ Dataset")
    st.write("""
    ข้อมูลที่ใช้มาจากเว็บไซต์ **www.kaggle.com**
    - Dataset: [Ramen Ratings](https://www.kaggle.com/datasets/residentmario/ramen-ratings)
    - Dataset นี้รวบรวมรีวิวราเมนจากทั่วโลกกว่า 2,580 รายการ พร้อมคะแนนรีวิว (`Stars`)
    """)

    st.header("🔹 Feature ของ Dataset")
    st.markdown("""
| Feature | คำอธิบาย |
|---|---|
| **Review #** | หมายเลขรีวิว |
| **Brand** | แบรนด์ของราเมน เช่น "Nissin", "Maruchan" |
| **Variety** | ชื่อของราเมน เช่น "Chicken Flavor", "Tonkotsu" |
| **Style** | รูปแบบ เช่น "Cup", "Pack", "Bowl" |
| **Country** | ประเทศที่ผลิต เช่น "Japan", "USA" |
| **Stars** | คะแนนรีวิว 0–5 (เป้าหมายของโมเดล) |
    """)

    st.header("🔹 การเตรียมข้อมูล (Data Preprocessing)")
    st.write("""
    Dataset ดั้งเดิมมีความไม่สมบูรณ์ในหลายจุด:
    - **ค่า Stars ที่ขาดหาย**: บางแถวไม่มีคะแนน → ลบทิ้ง (`dropna`)
    - **ค่า Stars เป็น "Unrated"**: ใช้ข้อความแทนตัวเลข → แปลงด้วย `pd.to_numeric` แล้วลบ Error
    - **Categorical Features**: Brand, Variety, Style, Country เป็น String → ใช้ **Label Encoding**
    - **Feature Scaling**: ใช้ **StandardScaler** ปรับค่าให้อยู่ใน scale เดียวกัน
    """)
    st.code("""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("ramen-ratings.csv")
df['Stars'] = pd.to_numeric(df['Stars'], errors='coerce')
df = df.dropna(subset=['Stars'])

label_encoders = {}
for col in ['Brand', 'Variety', 'Style', 'Country']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X = df[['Brand', 'Variety', 'Style', 'Country']].values
y = df['Stars'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
    """, language="python")

    st.header("🔹 ทฤษฎีของ Neural Network ที่ใช้")
    st.write("""
    โมเดลใช้ **MLPRegressor (Multi-Layer Perceptron)** จาก scikit-learn
    ซึ่งเป็น Fully Connected Neural Network สำหรับ **Regression Task**

    - **Input Layer**: รับ 4 Features
    - **Hidden Layers**: 3 ชั้น (256 → 128 → 64 neurons) ใช้ **ReLU Activation**
    - **Output Layer**: Neuron เดียว ทำนายค่าคะแนน Stars
    - **Optimizer**: **Adam** ปรับ Learning Rate อัตโนมัติ *(Kingma & Ba, 2014)*
    - **Early Stopping**: หยุด train อัตโนมัติเมื่อ validation loss ไม่ลดลง เพื่อป้องกัน Overfitting
    """)

    st.subheader("🔹 โครงสร้างของโมเดล")
    st.code("""
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),  # 3 Hidden Layers
    activation='relu',
    solver='adam',
    learning_rate_init=0.0005,
    max_iter=100,
    early_stopping=True,        # หยุดเมื่อ val_loss ไม่ดีขึ้น
    validation_fraction=0.1,
    random_state=42
)
    """, language="python")

    st.header("🔹 ขั้นตอนการพัฒนาโมเดล")
    st.write("""
    1. โหลดและทำความสะอาด Dataset
    2. ใช้ Label Encoding แปลง Categorical Features
    3. ใช้ StandardScaler ปรับค่าข้อมูล
    4. สร้าง MLPRegressor และ Train โมเดล
    5. วัดผลด้วย Mean Absolute Error (MAE)
    6. บันทึกโมเดลและ encoders ด้วย `joblib`
    """)
    st.code("""
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"MAE: {mae:.4f}")

joblib.dump(model, "models/ramen_model.pkl")
    """, language="python")

    st.header("📚 แหล่งอ้างอิง")
    st.write("""
    - Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.
    - Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization.* arXiv:1412.6980.
    - Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.
    - scikit-learn MLPRegressor Docs: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    - Kaggle Dataset: https://www.kaggle.com/datasets/residentmario/ramen-ratings
    """)

    st.success("✨ โมเดล Neural Network (MLP) พร้อมใช้งาน และนำไปทดสอบได้ที่หน้า Demo NN!")
