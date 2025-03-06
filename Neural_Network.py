import streamlit as st

st.title("Neural Network")

st.text("เกี่ยวกับ mnist | ข้อมูลตัวเลขที่เขียนด้วยลายมือ")
st.markdown("download มาจาก https://www.kaggle.com/datasets")

st.write("## อธิบาย feature ของ Dataset")
st.markdown("1.label - ค่าป้ายกำกับที่แทนตัวเลขที่เขียนด้วยลายมือ (ค่าระหว่าง 0-9)")
st.markdown("2.1x1 ถึง 28x28 - พิกเซลของภาพขนาด 28x28 พิกเซล (ค่าช่วง 0-255)")
st.markdown("ค่าที่ 0 หมายถึงพื้นหลังสีขาว , ค่าที่ 255 หมายถึงสีดำ , ค่าระหว่าง 0-255 หมายถึงระดับสีเทาของพิกเซลนั้น ๆ")

st.write("## การเตรียมข้อมูล")


st.write("## ทฤษฎีของ Linear Regression")



