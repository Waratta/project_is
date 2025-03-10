import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

# Set title of the app
st.title("Demo Machine Learning")

# โหลดข้อมูล
df = pd.read_csv("Flight_Price_Dataset_of_Bangladesh.csv")

# แสดงตัวอย่างข้อมูล
st.subheader("Sample Dataset")
st.write(df.head(10))

# แสดง dataset ทั้งหมดแบบแบ่งเป็นหน้าตาราง
st.subheader("Full Dataset")
st.dataframe(df)

# เลือก Feature ที่มีความสำคัญ
features = ["Duration (hrs)", "Stopovers", "Class", "Days Before Departure"]
target = "Total Fare (BDT)"

# แยกประเภทของ Features
numeric_features = ["Duration (hrs)", "Days Before Departure"]
categorical_features = ["Stopovers", "Class"]

# แปลงข้อมูล Categorical เป็นตัวเลข
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# แบ่งชุดข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# สร้างโมเดล Linear Regression
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

# สร้างโมเดล Random Forest
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

# ประเมินผล
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# แสดงผลลัพธ์ใน Streamlit
st.write(f"MAE Linear Regression: {mae_lr:.2f}")
st.write(f"MAE Random Forest: {mae_rf:.2f}")

# ลดจำนวนจุดข้อมูลในกราฟเพื่อให้อ่านง่ายขึ้น
sample_indices = np.linspace(0, len(y_test)-1, num=100, dtype=int)
y_test_sampled = y_test.values[sample_indices]
y_pred_lr_sampled = y_pred_lr[sample_indices]
y_pred_rf_sampled = y_pred_rf[sample_indices]

# แสดงกราฟเส้นแยกกัน
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(sample_indices, y_test_sampled, label="Actual Values", color='black', linestyle='dotted')
ax1.plot(sample_indices, y_pred_lr_sampled, label="Linear Regression Predictions", color='blue')
ax1.legend()
ax1.set_title("Actual vs Linear Regression Predictions")
ax1.set_xlabel("Sample Index")
ax1.set_ylabel("Total Fare (BDT)")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(sample_indices, y_test_sampled, label="Actual Values", color='black', linestyle='dotted')
ax2.plot(sample_indices, y_pred_rf_sampled, label="Random Forest Predictions", color='red')
ax2.legend()
ax2.set_title("Actual vs Random Forest Predictions")
ax2.set_xlabel("Sample Index")
ax2.set_ylabel("Total Fare (BDT)")
st.pyplot(fig2)