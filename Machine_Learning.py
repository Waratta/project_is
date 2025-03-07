import streamlit as st
import pandas as pd

st.title("Machine Learning")

st.text("เกี่ยวกับ California Housing Prices | ราคาบ้านในรัฐแคลิฟอเนีย")
st.markdown("download มาจาก https://www.kaggle.com/datasets")

st.write("## 📌อธิบาย feature ของ Dataset")

df = pd.DataFrame(
    {
        "คอลัมน์": ["longitude", "latitude"
                ,"housing_median_age","total_rooms","total_bedrooms","population",
                "households","median_income","median_house_value","ocean_proximity"],

        "คำอธิบาย": ["ค่าพิกัดลองจิจูดของพื้นที่ที่อยู่อาศัย", "ค่าพิกัดละติจูดของพื้นที่ที่อยู่อาศัย"
                ,"อายุเฉลี่ยของบ้านในบริเวณนั้น(หน่วยเป็นปี)","จำนวนห้องทั้งหมดในพื้นที่นั้น"
                ,"จำนวนห้องนอนทั้งหมดในพื้นที่นั้น","จำนวนประชากรที่อาศัยอยู่ในพื้นที่นั้น",
                "จำนวนครัวเรือนในพื้นที่นั้น","รายได้เฉลี่ยของครัวเรือน(วัดเป็นสัดส่วนจากค่าเฉลี่ยของสหรัฐ)"
                ,"มูลค่าบ้านเฉลี่ยในพื้นที่นั้น","ระยะห่างจากมหาสมุทร"]
    }
)
st.data_editor(
    df,
    column_config={
        "คอลัมน์":"คอลัมน์",
        "คำอธิบาย": st.column_config.TextColumn(
            "คำอธิบาย",
            default="st.",
            max_chars=50,
            validate=r"^st\.[a-z_]+$",
        )
    },
    hide_index=True,
)

st.write("## 📌การเตรียมข้อมูล")
st.markdown("1. Pipeline สำหรับตัวแปรเชิงตัวเลข ใช้ค่ามัธยฐาน(median) เติมค่าที่หายไป เเละใช้ StandardScaler() ทำ Feature Scaling")

code = '''# Pipeline สำหรับตัวแปรเชิงตัวเลข
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])'''
st.code(code, language="python")

st.markdown("2. แปลงข้อมูลประเภทหมวดหมู่ เนื่องจาก ocean_proximity เป็นข้อมูลแบบ หมวดหมู่ (Categorical) เราต้องแปลงให้เป็นตัวเลข โดยใช้ One-Hot Encoding")

code = '''# Pipeline สำหรับตัวแปรหมวดหมู่
cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])'''
st.code(code, language="python")

st.markdown("3. รวมทั้งสอง Pipeline ใช้ ColumnTransformer() รวมทั้ง Numerical Pipeline และ Categorical Pipeline")

code = '''# รวมทุก Pipeline
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])'''
st.code(code, language="python")


st.write("## 📌ทฤษฎีของ Linear Regression")
st.markdown("เป็น Machine Learning ประเภท Supervised Learning โดยต้องใส่ dataset ก่อน "
"เเล้วโปรแกรมจะนำตัวแปรต้นและตัวแปรตามไปคำนวณด้วยสถิติทางคณิตศาสตร์ เเละได้ข้อมูลกลับมาเป็นตัวเลข"
"โดยสมการความสัมพันธ์ของ Linear Regression ก็คือ y=mx+c เมื่อ ")

col1, col2 = st.columns(2)

with col1:
    st.markdown("x คือ ตัวแปรต้น ")
    st.markdown("y คือ ตัวแปรตาม ")

with col2:
    st.markdown("m คือ ความชัน ")
    st.markdown("c คือ จุดตัดแกน y ")




st.write("## 📌ทฤษฎีของ Random Forest Regressor")
st.markdown("หลักการของ Random Forest คือ สร้าง model จาก Decision Tree หลายๆ model ย่อยๆ "
"เเต่ละ model ได้ data set ไม่เหมือนกัน ตอนทำ prediction ก็ให้แต่ละ Decision Tree ทำ prediction ของใครของมัน "
"และคำนวณผล prediction มาจากค่า vote มากที่สุดของ Decision Tree  หรือ หาค่า mean จาก output ของแต่ละ Decision Tree "
"Decision Tree แต่ละ model ใน Random Forest ถือว่าเป็น weak learner เพราะเป็น model ที่ไม่เก่งเท่าไหร่  "
"แต่พอมาทำ prediction ร่วมกัน ก็จะได้ model ที่มีความเเม่นยำมากกว่า")

st.write("## 📌ขั้นตอนการพัฒนาโมเดล")
st.markdown("1. แบ่งข้อมูล Train & Test")

code = '''from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)'''
st.code(code, language="python")

st.markdown("2. เทรนและประเมินผลโมเดล สร้าง Dictionary ที่เก็บโมเดล Linear Regression และ Random Forest")
code = '''# สร้างโมเดลและประเมินผล
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
}'''
st.code(code, language="python")

st.markdown("✅ วนลูปเทรนโมเดล\n"
            "1. สร้าง Pipeline (preprocessor + regressor)\n"
            "2. เทรนโมเดลด้วย .fit()\n"
            "3. ทำนายค่าด้วย .predict()\n"
            "4. คำนวณ Mean Absolute Error (MAE)\n"
            "5. คำนวณ Root Mean Squared Error (RMSE)")

code = '''results = []
feature_importance = {}

for model_name, model in models.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append({"Model": model_name, "MAE": mae, "RMSE": rmse})
    
    if hasattr(model, 'feature_importances_'):
        feature_importance[model_name] = model.feature_importances_
'''
st.code(code, language="python")

st.markdown("3. แสดงผลลัพธ์ใน Streamlit แสดง ผลลัพธ์ของโมเดล (MAE & RMSE) ในรูปแบบ DataFrame ใช้ Seaborn Bar Chart เปรียบเทียบโมเดล")
code = '''# แสดงผลลัพธ์ใน Streamlit
results_df = pd.DataFrame(results)
st.write("## ผลลัพธ์ของโมเดล")
st.dataframe(results_df)

# วาดกราฟเปรียบเทียบ MAE และ RMSE
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# กราฟ MAE
sns.barplot(x="Model", y="MAE", data=results_df, ax=ax[0], palette="Blues_r")
ax[0].set_title("Mean Absolute Error (MAE)")
ax[0].set_ylabel("MAE")
ax[0].set_xlabel("Model")

# กราฟ RMSE
sns.barplot(x="Model", y="RMSE", data=results_df, ax=ax[1], palette="Reds_r")
ax[1].set_title("Root Mean Squared Error (RMSE)")
ax[1].set_ylabel("RMSE")
ax[1].set_xlabel("Model")

st.pyplot(fig)
'''
st.code(code, language="python")