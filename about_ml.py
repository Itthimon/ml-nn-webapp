import streamlit as st

def show():
    st.title("📖 การพัฒนาโมเดล Machine Learning (Pokedex)")

    # แหล่งที่มาของ Dataset
    st.header("📌 แหล่งที่มาของ Dataset")
    st.write("""
    ข้อมูลที่ใช้มาจากเว็บไซต์ **www.kaggle.com**
    - Dataset: [Complete Pokemon Dataset](https://www.kaggle.com/datasets/mrdew25/pokemon-database)
    - Dataset นี้รวบรวมข้อมูลโปเกมอนทั้งหมด 1,025 ตัว รวมถึงค่าสถิติพื้นฐาน เช่น HP, Attack, Defense และประเภท
    """)

    # Feature ของ Dataset
    st.header("🔹 Feature ของ Dataset")
    st.markdown("""
| Feature | คำอธิบาย |
|---|---|
| **id** | หมายเลขโปเกมอนใน Pokedex |
| **name** | ชื่อของโปเกมอน เช่น "Pikachu", "Charizard" |
| **type** | ประเภทของโปเกมอน (อาจมี 1 หรือ 2 ประเภท) |
| **hp** | ค่าพลังชีวิต |
| **attack** | ค่าพลังโจมตีทางกายภาพ |
| **defense** | ค่าพลังป้องกันทางกายภาพ |
| **s_attack** | ค่าพลังโจมตีพิเศษ |
| **s_defense** | ค่าพลังป้องกันพิเศษ |
| **speed** | ค่าความเร็ว |
| **height / weight** | ส่วนสูงและน้ำหนักของโปเกมอน |
    """)

    # การเตรียมข้อมูล
    st.header("🔹 การเตรียมข้อมูล (Data Preprocessing)")
    st.write("""
    Dataset ดั้งเดิม (`pokedex.csv`) มีความไม่สมบูรณ์ในหลายจุด ได้แก่:

    - **ข้อมูล weight ผิดปกติ**: มีโปเกมอนบางตัวที่มีน้ำหนักเกิน 5,000 → ตัดทิ้ง
    - **รูปแบบ type ที่ซับซ้อน**: เก็บในรูปแบบ `{grass,poison}` → แยก primary type ออกมาด้วย Regex
    - **ข้อมูลหลาย type**: โปเกมอนที่มี 2 ประเภท → ใช้ประเภทแรกเป็น Target Label
    - **Feature Scaling**: ค่าสถิติโปเกมอนมีช่วงกว้างต่างกัน → ใช้ `StandardScaler` ปรับให้อยู่ใน scale เดียวกัน
    - **Label Encoding**: แปลงชื่อประเภท (เช่น "fire", "water") เป็นตัวเลขด้วย `LabelEncoder`
    """)

    st.subheader("🔹 โค้ดการเตรียมข้อมูล")
    st.code("""
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("pokedex.csv")

# ลบข้อมูล weight ที่ผิดปกติ
df = df[df['weight'] <= 5000]

# แยก primary type
def extract_primary_type(t):
    types = re.findall(r'[a-z]+', str(t).lower())
    return types[0] if types else 'unknown'

df['primary_type'] = df['type'].apply(extract_primary_type)
df = df[df['primary_type'] != 'unknown']

# Features และ Target
features = ['attack', 'defense', 'hp', 'speed', 's_attack', 's_defense']
X = df[features].values
y = LabelEncoder().fit_transform(df['primary_type'].values)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
    """, language="python")

    # ทฤษฎีของ Ensemble Model
    st.header("🔹 ทฤษฎีของ Ensemble Learning")
    st.write("""
    **Ensemble Learning** คือแนวทางการรวมโมเดลหลายตัวเข้าด้วยกัน เพื่อให้ผลการพยากรณ์ดีกว่าการใช้โมเดลตัวเดียว
    โดยแนวคิดพื้นฐานคือ "ฝูงชนที่ฉลาดมักตัดสินใจดีกว่าคนเดียว" *(Dietterich, 2000)*

    โปรเจคนี้ใช้ **Voting Classifier** ซึ่งรวม 3 โมเดลที่แตกต่างกัน:
    """)
    st.markdown("""
| โมเดล | หลักการ |
|---|---|
| **Random Forest (RF)** | สร้าง Decision Tree จำนวนมากแบบสุ่ม แล้วโหวตผล |
| **Gradient Boosting (GB)** | สร้างโมเดลแบบต่อเนื่อง โดยแต่ละตัวแก้ข้อผิดพลาดของตัวก่อน |
| **K-Nearest Neighbors (KNN)** | พยากรณ์จาก k ตัวอย่างที่ใกล้ที่สุดในพื้นที่ Feature |
    """)
    st.write("""
    ใช้ **Soft Voting** → แต่ละโมเดลส่ง **ความน่าจะเป็น** มาเฉลี่ยกัน ให้ผลแม่นกว่า Hard Voting ที่แค่นับเสียงข้างมาก
    """)

    # ขั้นตอนการพัฒนาโมเดล
    st.header("🔹 ขั้นตอนการพัฒนาโมเดล")
    st.write("""
    1. โหลดและทำความสะอาด Dataset
    2. เตรียม Feature ด้วย StandardScaler
    3. สร้าง VotingClassifier จาก RF + GB + KNN
    4. Train ด้วย `model.fit(X_train, y_train)`
    5. วัดผลด้วย Accuracy บน Test Set
    6. บันทึกโมเดล, Scaler, LabelEncoder ด้วย `joblib`
    """)

    st.subheader("🔹 โค้ดการสร้างและ Train โมเดล")
    st.code("""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

rf  = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
gb  = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('knn', knn)],
    voting='soft'   # ใช้ค่า probability เฉลี่ย ไม่ใช่แค่โหวต
)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
ensemble.fit(X_train, y_train)

# บันทึกโมเดล
joblib.dump(ensemble, "models/pokedex_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")
    """, language="python")

    # แหล่งอ้างอิง
    st.header("📚 แหล่งอ้างอิง")
    st.write("""
    - Dietterich, T. G. (2000). *Ensemble Methods in Machine Learning.* Multiple Classifier Systems, Lecture Notes in Computer Science.
    - Breiman, L. (2001). *Random Forests.* Machine Learning, 45, 5–32.
    - Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics, 29(5), 1189–1232.
    - Cover, T., & Hart, P. (1967). *Nearest Neighbor Pattern Classification.* IEEE Transactions on Information Theory, 13(1), 21–27.
    - scikit-learn Documentation: https://scikit-learn.org/stable/modules/ensemble.html
    - Kaggle Dataset: https://www.kaggle.com/datasets/mrdew25/pokemon-database
    """)

    st.success("✨ โมเดล Ensemble (RF + GB + KNN) พร้อมใช้งาน และนำไปทดสอบได้ที่หน้า Demo ML!")
