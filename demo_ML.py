import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ตั้งค่าธีมของ Seaborn
sns.set_style("whitegrid")

st.title("Demo Machine Learning Analysis for Housing Data")

# โหลดชุดข้อมูล
df = pd.read_csv("housing.csv/housing.csv")

# แสดงข้อมูลตัวอย่าง
st.write("## ข้อมูลตัวอย่าง")
st.dataframe(df)

# กำหนดตัวแปรเป้าหมาย
target = "median_house_value"
X = df.drop(columns=[target])
y = df[target]

# แยกประเภทตัวแปร
num_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# ดูความสัมพันธ์ระหว่างตัวแปร (เฉพาะตัวเลข)
st.write("## Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df[num_features].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)

# Pipeline สำหรับตัวแปรเชิงตัวเลข
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Pipeline สำหรับตัวแปรหมวดหมู่
cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# รวมทุก Pipeline
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# แบ่งข้อมูลเป็นชุด Train & Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดลและประเมินผล
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = []
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

# แสดงผลลัพธ์ใน Streamlit
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

# แสดง Feature Importance ถ้ามี
if feature_importance:
    st.write("## Feature Importance")
    for model_name, importance in feature_importance.items():
        feature_names = num_features + list(preprocessor.named_transformers_["cat"].get_feature_names_out())
        feature_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        feature_df = feature_df.sort_values(by="Importance", ascending=False)
        
        st.write(f"### {model_name}")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Importance", y="Feature", data=feature_df, palette="viridis", ax=ax)
        ax.set_title(f"Feature Importance - {model_name}")
        st.pyplot(fig)
