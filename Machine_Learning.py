import streamlit as st

st.title("Machine Learning")

st.text("เกี่ยวกับ California Housing Prices | ราคาบ้านในรัฐแคลิฟอเนีย")
st.markdown("download มาจาก https://www.kaggle.com/datasets")

st.write("## อธิบาย feature ของ Dataset")
st.markdown("1.longitude - ค่าพิกัดลองจิจูดของพื้นที่ที่อยู่อาศัย")
st.markdown("2.atitude - ค่าพิกัดละติจูดของพื้นที่ที่อยู่อาศัย")
st.markdown("3.housing_median_age - อายุเฉลี่ยของบ้านในบริเวณนั้น (หน่วยเป็นปี)")
st.markdown("4.total_rooms - จำนวนห้องทั้งหมดในพื้นที่นั้น")
st.markdown("5.total_bedrooms - จำนวนห้องนอนทั้งหมดในพื้นที่นั้น")
st.markdown("6.population - จำนวนประชากรที่อาศัยอยู่ในพื้นที่นั้น")
st.markdown("7.households - จำนวนครัวเรือนในพื้นที่นั้น")
st.markdown("8.median_income - รายได้เฉลี่ยของครัวเรือน (วัดเป็นสัดส่วนจากค่าเฉลี่ยของสหรัฐ)")
st.markdown("9.median_house_value - มูลค่าบ้านเฉลี่ยในพื้นที่นั้น")
st.markdown("10.ocean_proximity - ประเภทของพื้นที่ตามระยะห่างจากชายฝั่ง")

st.write("## การเตรียมข้อมูล")
st.markdown("ก่อนที่เราจะพัฒนาโมเดล จำเป็นต้องทำความสะอาดและเตรียมข้อมูลให้พร้อมใช้งาน:")
st.markdown("1.จัดการค่าที่หายไป (Missing Values) → คอลัมน์ total_bedrooms มีค่าหายไป 207 ค่า สามารถแก้ไขได้โดยการเติมค่ามัธยฐาน (median)")
st.markdown("2.แปลงข้อมูลประเภทหมวดหมู่ เนื่องจาก ocean_proximity เป็นข้อมูลแบบ หมวดหมู่ (Categorical) เราต้องแปลงให้เป็น ตัวเลข โดยใช้ One-Hot Encoding")
st.markdown("3.การปรับขนาดข้อมูล (Feature Scaling) บางคอลัมน์ เช่น total_rooms และ population มีค่าตัวเลขขนาดใหญ่ ซึ่งอาจทำให้โมเดล เรียนรู้ได้ไม่ดี เราใช้ Standardization (ค่าเฉลี่ย 0, ส่วนเบี่ยงเบนมาตรฐาน 1)")

st.write("## ทฤษฎีของ Linear Regression")
st.markdown("เป็น Machine Learning ประเภท Supervised Learning โดยเราต้องใส่ dataset ก่อน "
"โดยโปรแกรมจะนำตัวแปรต้นและตัวแปรตามไปคำนวณด้วยสถิติทางคณิตศาสตร์ แล้วก็จะได้ข้อมูลกลับมาเป็นตัวเลข"
"โดยสมการความสัมพันธ์ของ Linear Regression ก็คือ y=mx+c เมื่อ ")
st.markdown("x คือ ตัวแปรต้น ")
st.markdown("y คือ ตัวแปรตาม ")
st.markdown("m คือ ความชัน ")
st.markdown("c คือ จุดตัดแกน y ")

st.write("## ทฤษฎีของ Random Forest Regressor")
st.markdown("Random Forest เป็น Ensemble Learning ที่ใช้หลายๆ Decision Trees มารวมกันเพื่อลด Overfitting โดยการสุ่มเลือกตัวแปรและข้อมูลฝึกสอน")

st.write("## ขั้นตอนการพัฒนาโมเดล")
st.markdown("1.แบ่งข้อมูล Train & Test")
st.markdown("2.การสร้าง Pipeline ใช้ Pipeline รวมขั้นตอน Preprocessing + Model Training")
st.markdown("3.เทรนโมเดล")
st.markdown("4.ประเมินผลโมเดล ใช้ค่า MAE, RMSE วัดความแม่นยำของโมเดล")