import streamlit as st

st.title("Neural Network")

st.text("เกี่ยวกับ mnist | ข้อมูลตัวเลขที่เขียนด้วยลายมือ")
st.markdown("download มาจาก https://www.kaggle.com/datasets")

st.write("## 📌อธิบาย feature ของ Dataset")
st.markdown("1.label - ค่าป้ายกำกับที่แทนตัวเลขที่เขียนด้วยลายมือ (ค่าระหว่าง 0-9)")
st.markdown("2.1x1 ถึง 28x28 - พิกเซลของภาพขนาด 28x28 พิกเซล (ค่าช่วง 0-255)")
st.markdown("ค่าที่ 0 หมายถึงพื้นหลังสีขาว , ค่าที่ 255 หมายถึงสีดำ , ค่าระหว่าง 0-255 หมายถึงระดับสีเทาของพิกเซลนั้น ๆ")

st.write("## 📌การเตรียมข้อมูล")
st.markdown("1.แปลงข้อมูลจากภาพเป็น csv ภาพตัวเลขขนาด 28x28 พิกเซล ที่ถูกแปลงเป็นรูปแบบตาราง (csv) โดยแต่ละแถวในตารางมี: "
"คอลัมน์ label → ค่าตัวเลขที่แท้จริง (0-9)"
"784 คอลัมน์ (pixel1 ถึง pixel784) → ค่าความเข้มของพิกเซลตั้งแต่ 0-255")
st.markdown("2.โหลดข้อมูล ใช้ pandas โหลดข้อมูลจากไฟล์ CSV และแยกตัวแปรอิสระ (X) และตัวแปรเป้าหมาย (y)")

code = '''# Load dataset
file_path = "mnist_test.csv"
df = pd.read_csv(file_path)

# Split features and labels
y = df["label"].values
X = df.drop("label", axis=1).values / 255.0  # Normalize pixel values'''
st.code(code, language="python")

st.markdown("3.การปรับแต่งข้อมูล (Data Preprocessing) "
"ใช้วิธี Normalization "
"ค่า Pixel แต่ละค่ามีช่วง 0-255"
"แปลงให้เป็นช่วง 0-1 โดยหารด้วย 255 เพื่อช่วยให้โมเดลเรียนรู้ได้เร็วขึ้นและลด Overfitting")

code = '''X = X / 255.0'''
st.code(code, language="python")

st.write("## 📌ทฤษฎีของ Multilayer Perceptron (MLP)")
st.markdown("รูปแบบของ Neural Network จะแบ่ง Perceptron ออกเป็นชั้น โดยแต่ละชั้นจะเรียกเป็น Layer "
"โดยข้อมูลที่เข้ามาจะไหลไปในทิศทางเดียว ไม่ไหลย้อนกลับจาก Layer นึงสู่อีก Layer นึง "
"โครงสร้างของ Neural Network แบบเพอร์เซ็ปตรอนหลายชั้น(Multi-Layer Perceptron : MLP)"
"จะประสานการทำงานผ่านส่วนที่เรียกว่าโหนด (Node) ประกอบไปด้วย 3 ชั้นมีการเชื่อมต่อการทำงานของโหนดอย่างสมบูรณ์ (Fully-Connected) ได้แก่")
st.markdown("1.Input Layer ชั้นนี้จะเป็นส่วนที่จัดการข้อมูล input จำนวนของโหนดขึ้นอยู่กับจำนวนของ input "
"ว่าข้อมูลอะไรบ้างที่จะนำเข้ามาคิดในโมเดล (ใน ML เรียกส่วนนี้ว่า Feature)")
st.markdown("2.Hidden Layer ชั้นที่อยู่ตรงกลางมีผลต่อประสิทธิภาพในการเรียนรู้ของโมเดล ซึ่งชั้นนี้จะมีกี่ชั้นก็ได้ "
"มีจำนวนของ Neuron เท่าไหร่ก็ได้ การเพิ่มชั้นและจำนวน neuron ก็จะส่งผลต่อการทำงานของโมเดลมากๆ มีการเรียนรู้ข้อมูลในเชิงลึกเรียกว่า Deep Learning")
st.markdown("3.Output Layer ชั้นนี้จะนำเอาข้อมูลจากการคำนวณใน Hidden Layer "
"ไปใช้งานจำนวนของโหนดที่ส่งออกมาในชั้นนี้ขึ้นอยู่กับรูปแบบของ output ที่จะเอาไปใช้")

st.write("## 📌ขั้นตอนการพัฒนาโมเดล")
st.markdown("1.การสร้างโมเดล (Model Architecture) ฟังก์ชัน create_model() ใช้ TensorFlow/Keras สร้างโมเดลตามโครงสร้างที่กล่าวไว้")

code = '''# Define model
def create_model():
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model'''
st.code(code, language="python")

st.markdown("2.แสดงโครงสร้างของโมเดล ใช้ model.summary() เพื่อแสดงรายละเอียดของโมเดล เช่น จำนวนพารามิเตอร์และ Layer ต่างๆ")

code = '''# Display model summary
st.subheader("Model Summary")
string_io = io.StringIO()
model.summary(print_fn=lambda x: string_io.write(x + "\n"))
st.text(string_io.getvalue())'''
st.code(code, language="python")
st.markdown("✅ ช่วยให้เห็นว่าโมเดลมี Layer อะไรบ้าง และมีพารามิเตอร์ที่ต้องเรียนรู้กี่ตัว")

st.markdown("3.การฝึกโมเดล (Model Training) กดปุ่ม 'Train Model' เพื่อเริ่มการฝึก "
"ใช้ model.fit() เทรนโมเดล 5 รอบ (epochs=5) "
"ใช้ batch_size=32 เพื่ออัปเดตค่าถ่วงน้ำหนักทุก ๆ 32 ตัวอย่าง "
"ใช้ validation_split=0.1 เพื่อแบ่งข้อมูล 10% สำหรับตรวจสอบความถูกต้องของโมเดล")

code = '''if st.button("Train Model"):
    with st.spinner("Training in progress..."):
        history = model.fit(X, y, epochs=5, batch_size=32, validation_split=0.1, verbose=0)
    st.success("Training complete!")'''
st.code(code, language="python")

st.markdown("4.แสดงผลการฝึกโมเดล "
"history.history['accuracy'] → ค่าความแม่นยำของชุด Train "
"history.history['val_accuracy'] → ค่าความแม่นยำของชุด Validation")

code = '''# Display results
    st.write("Final Training Accuracy:", history.history['accuracy'][-1])
    st.write("Final Validation Accuracy:", history.history['val_accuracy'][-1])'''
st.code(code, language="python")

st.markdown("5.แสดงกราฟผลการฝึก (Training Visualization) "
"ใช้ Matplotlib วาดกราฟ:"
"1.Accuracy (ความแม่นยำของโมเดล)"
"2.Loss (ค่าความผิดพลาด)")

code = '''# Plot training history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    ax[0].plot(history.history['accuracy'], label='Training Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title("Model Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    
    ax[1].plot(history.history['loss'], label='Training Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].set_title("Model Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    
    st.pyplot(fig)'''
st.code(code, language="python")

