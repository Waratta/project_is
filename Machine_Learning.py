import streamlit as st
import pandas as pd

st.title("Machine Learning")
st.markdown("<p style='text-align: right;'>สมาชิก 1 : นางสาววรัชฐญา จั่นเล็ก 6604062610535</p>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align: right;'>สมาชิก 2 : นางสาวสุชานันท์ ปิ่นทอง 6604062610578</p>",
            unsafe_allow_html=True)

st.text("เกี่ยวกับ Flight Price Dataset of Bangladesh | ราคาตั๋วเครื่องบินในบังคลาเทศ")
st.markdown("download มาจาก https://www.kaggle.com/datasets/mahatiratusher/flight-price-dataset-of-bangladesh")

st.write("## 📌อธิบาย feature ของ Dataset")
st.markdown("### 1. Categorical Features (ข้อมูลหมวดหมู่)")
st.markdown("ข้อมูลประเภทหมวดหมู่จะถูกเข้ารหัสเป็นตัวเลขโดยใช้ OneHotEncoder ในขั้นตอนการพัฒนาโมเดล")
df = pd.DataFrame(
    {
        "Feature": ["Airline", "Source"
                ,"Source Name","Destination","Destination Name","Stopovers",
                "Aircraft Type","Class","Booking Source","Seasonality"],

        "ความหมาย": ["ชื่อสายการบิน", "รหัสสนามบินต้นทาง"
                ,"ชื่อสนามบินต้นทาง","รหัสสนามบินปลายทาง"
                ,"ชื่อสนามบินปลายทาง","จำนวนจุดแวะพัก","ประเภทของเครื่องบิน","ชั้นโดยสารของตั๋ว"
                ,"ช่องทางการจอง","ฤดูกาลของเที่ยวบิน"]
    }
)
st.data_editor(
    df,
    column_config={
        "Feature":"Feature",
        "ความหมาย": st.column_config.TextColumn(
            "ความหมาย",
            default="st.",
            max_chars=50,
            validate=r"^st\.[a-z_]+$",
        )
    },
    hide_index=True,
)
st.markdown("### 2. Numerical Features (ข้อมูลเชิงตัวเลข)")
st.markdown("ข้อมูลประเภทตัวเลขจะถูกปรับขนาดโดยใช้ StandardScaler เพื่อให้ข้อมูลอยู่ในช่วงเดียวกัน")
df = pd.DataFrame(
    {
        "Feature": ["Duration (hrs)", "Base Fare (BDT)"
                ,"Tax & Surcharge (BDT)","Total Fare (BDT)","Days Before Departure"],

        "ความหมาย": ["ระยะเวลาในการเดินทาง (ชั่วโมง)", "ราคาตั๋วพื้นฐาน (ก่อนบวกภาษีและค่าธรรมเนียม)"
                ,"ภาษีและค่าธรรมเนียม","ราคารวมหลังจากรวมภาษีและค่าธรรมเนียม"
                ,"	จำนวนวันก่อนการเดินทาง"]
    }
)
st.data_editor(
    df,
    column_config={
        "Feature":"Feature",
        "ความหมาย": st.column_config.TextColumn(
            "ความหมาย",
            default="st.",
            max_chars=50,
            validate=r"^st\.[a-z_]+$",
        )
    },
    hide_index=True,
)

st.markdown("### 3. Date-Time Features (ข้อมูลประเภทวันและเวลา)")
st.markdown("ข้อมูลประเภทเวลาอาจถูกแปลงเป็น Feature ใหม่ เช่น ช่วงเวลา หรือ วันในสัปดาห์")
df = pd.DataFrame(
    {
        "Feature": ["Departure Date & Time", "Arrival Date & Time"],

        "ความหมาย": ["วันและเวลาที่ออกเดินทาง", "วันและเวลาที่มาถึง"]
    }
)
st.data_editor(
    df,
    column_config={
        "Feature":"Feature",
        "ความหมาย": st.column_config.TextColumn(
            "ความหมาย",
            default="st.",
            max_chars=50,
            validate=r"^st\.[a-z_]+$",
        )
    },
    hide_index=True,
)

st.write("## 📌การเตรียมข้อมูล")
st.markdown("1. เลือก Features ที่มีความสำคัญ")
st.markdown("✅ เหตุผลในการเลือก Features :")
st.markdown("- Duration (hrs) → ระยะเวลาการเดินทางมีผลต่อราคา\n"
"\n- Stopovers → จำนวนจุดแวะพักส่งผลต่อความสะดวกสบายและราคา\n"
"\n- Class → ชั้นโดยสารส่งผลต่อราคาตั๋ว\n"
"\n- Days Before Departure → ระยะห่างก่อนการเดินทางมักส่งผลต่อราคา (ยิ่งใกล้วันเดินทาง ราคามักสูงขึ้น)")

code = '''# เลือก Feature ที่มีความสำคัญ
features = ["Duration (hrs)", "Stopovers", "Class", "Days Before Departure"]
target = "Total Fare (BDT)"'''
st.code(code, language="python")

st.markdown("2. จัดการประเภทข้อมูล (Categorical & Numerical)")
st.markdown("ในโค้ด ได้แยกประเภทข้อมูลออกเป็น :")
st.markdown("- Categorical Features : Stopovers , Class")
st.markdown("- Numeric Features : Duration (hrs) , Days Before Departure")

code = '''# แยกประเภทของ Features
numeric_features = ["Duration (hrs)", "Days Before Departure"]
categorical_features = ["Stopovers", "Class"]'''
st.code(code, language="python")

st.markdown("3. การทำ Preprocessing ด้วย ColumnTransformer")
st.markdown("ในโค้ดใช้ ColumnTransformer เพื่อจัดการกับข้อมูลประเภทต่าง ๆ :\n"
"\n- ใช้ StandardScaler เพื่อปรับขนาดข้อมูลเชิงตัวเลขให้อยู่ในช่วงเดียวกัน\n"
"\n- ใช้ OneHotEncoder เพื่อแปลงข้อมูลเชิงหมวดหมู่ให้กลายเป็นตัวเลข (0, 1)")

code = '''# แปลงข้อมูล Categorical เป็นตัวเลข
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
])'''
st.code(code, language="python")

st.markdown("4. แบ่งข้อมูลเป็น Train/Test")
st.markdown("แบ่งข้อมูลออกเป็นชุด Train และ Test โดยใช้ train_test_split:\n"
"\n- Train Set: 80% → ใช้สำหรับ Train โมเดล\n"
"\n- Test Set: 20% → ใช้สำหรับประเมินผล")

code = '''# แบ่งชุดข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
'''
st.code(code, language="python")

st.write("## 📌ทฤษฎีของ Linear Regression")
st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;เป็น Machine Learning ประเภท Supervised Learning ต้องมี dataset ที่มีตัวแปรต้นและตัวแปรตาม "
"โปรแกรมจะใช้หลักสถิติทางคณิตศาสตร์เพื่อสร้างสมการความสัมพันธ์ระหว่างตัวแปรต้นและตัวแปรตาม")
st.markdown("สมการของ Linear Regression คือ : y = mx + c")

col1, col2 = st.columns(2)

with col1:
    st.markdown("- x คือ ตัวแปรต้น(Input) ")
    st.markdown("- y คือ ตัวแปรตาม(Output) ")

with col2:
    st.markdown("- m คือ ความชัน ")
    st.markdown("- c คือ จุดตัดแกน y ")

st.write("## 📌ทฤษฎีของ Random Forest Regressor")
st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Random Forest = รวมหลาย Decision Tree ที่ใช้ dataset ต่างกัน"
"ทำ prediction โดยให้แต่ละ Tree ทำนาย แล้วรวมผลโดย:\n"
"- Classification → ใช้ค่า vote มากสุด\n "
"- Regression → ใช้ค่า mean ของผลลัพธ์\n"
"\nแต่ละ Decision Tree เป็น weak learner → รวมกันแล้วได้ model ที่แม่นยำขึ้น")

st.write("## 📌ขั้นตอนการพัฒนาโมเดล")
st.markdown("1. การ Train & Test โมเดลทั้ง Linear Regression เเละ Random Forest")
st.markdown("ใช้ fit() เพื่อสอนโมเดลด้วยข้อมูล Training Set :")
st.markdown("ใช้ predict() เพื่อทำนายข้อมูลใน Test Set :")

code = '''# สร้างโมเดล Linear Regression
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)'''
st.code(code, language="python")

code = '''# สร้างโมเดล Random Forest
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)'''
st.code(code, language="python")

st.markdown("2. การประเมินผล (Evaluation)")
st.markdown("ใช้ Mean Absolute Error (MAE) เพื่อวัดความแม่นยำของโมเดล :")

code = '''# ประเมินผล
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_rf = mean_absolute_error(y_test, y_pred_rf)'''
st.code(code, language="python")


st.markdown("3. การแสดงผล (Visualization)")
st.markdown("ในโค้ดมีการใช้ matplotlib เพื่อแสดงกราฟเปรียบเทียบค่าที่ทำนายกับค่าจริง :")
st.markdown("โค้ดกราฟ Linear Regression")

code = '''fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(sample_indices, y_test_sampled, label="Actual Values", color='black', linestyle='dotted')
ax1.plot(sample_indices, y_pred_lr_sampled, label="Linear Regression Predictions", color='blue')
ax1.legend()
ax1.set_title("Actual vs Linear Regression Predictions")
ax1.set_xlabel("Sample Index")
ax1.set_ylabel("Total Fare (BDT)")
st.pyplot(fig1)
'''
st.code(code, language="python")

st.markdown("โค้ดกราฟ Random Forest")

code = '''fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(sample_indices, y_test_sampled, label="Actual Values", color='black', linestyle='dotted')
ax2.plot(sample_indices, y_pred_rf_sampled, label="Random Forest Predictions", color='red')
ax2.legend()
ax2.set_title("Actual vs Random Forest Predictions")
ax2.set_xlabel("Sample Index")
ax2.set_ylabel("Total Fare (BDT)")
st.pyplot(fig2)
'''
st.code(code, language="python")

st.markdown("<p style='text-align: right;'>อ้างอิงจาก chatGPT</p>",
            unsafe_allow_html=True)