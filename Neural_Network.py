import streamlit as st
import pandas as pd

st.title("Neural Network")

st.text("เกี่ยวกับ mnist | ข้อมูลตัวเลขที่เขียนด้วยลายมือ")
st.markdown("download มาจาก https://www.kaggle.com/datasets/oddrationale/mnist-in-csv")

st.write("## 📌อธิบาย feature ของ Dataset")

df =pd.DataFrame( {
    'ข้อมูล': ['Label', '1x1 ถึง 28x28'],
    'รายละเอียด': [
        'ค่าป้ายกำกับแทนตัวเลขที่เขียนด้วยลายมือ (0-9)',
        'พิกเซลของภาพขนาด 28x28 (ค่า 0-255)'
    ]
}
)
st.data_editor(
    df,
    column_config={
        "ข้อมูล":"ข้อมูล",
        "รายละเอียด": st.column_config.TextColumn(
            "คำอธิบาย",
            default="st.",
            max_chars=50,
            validate=r"^st\.[a-z_]+$",
        )
    },
    hide_index=True,
)

st.write("## 📌การเตรียมข้อมูล")
st.markdown("1.แปลงข้อมูลจากภาพเป็น csv ภาพตัวเลขขนาด 28x28 พิกเซล ที่ถูกแปลงเป็นรูปแบบตาราง (csv) โดยแต่ละแถวในตารางมี :\n "
"\n- คอลัมน์ label → ค่าตัวเลขที่แท้จริง (0-9)\n"
"\n- 784 คอลัมน์ (pixel1 ถึง pixel784) → ค่าความเข้มของพิกเซลตั้งแต่ 0-255\n")


st.markdown("2.โหลดข้อมูล ใช้ pandas โหลดข้อมูลจากไฟล์ CSV และแยกตัวแปรอิสระ (X) และตัวแปรเป้าหมาย (y)")

code = '''# Load dataset
file_path = "mnist_test.csv"
df = pd.read_csv(file_path)

# Split features and labels
y = df["label"].values
X = df.drop("label", axis=1).values / 255.0  # Normalize pixel values'''
st.code(code, language="python")

st.markdown("3. การปรับแต่งข้อมูล (Data Preprocessing)  ใช้วิธี Normalization\n "
"\n- ค่า Pixel แต่ละค่ามีช่วง 0-255\n"
"\n- แปลงให้เป็นช่วง 0-1 โดยหารด้วย 255 เพื่อช่วยให้โมเดลเรียนรู้ได้เร็วขึ้นและลด Overfitting")

code = '''X = X / 255.0'''
st.code(code, language="python")

st.write("## 📌ทฤษฎีของ Multilayer Perceptron (MLP)")
st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Neural Network แบ่งโครงสร้างออกเป็น Layer โดยข้อมูลจะไหลในทิศทางเดียว (Feedforward)\n"
"\nMulti-Layer Perceptron (MLP) → เชื่อมต่อโหนดแบบสมบูรณ์ (Fully-Connected)")

st.markdown("- Input Layer → รับข้อมูลนำเข้า (Feature)")
st.markdown("- Hidden Layer → อยู่ตรงกลาง มีผลต่อประสิทธิภาพของโมเดล (ปรับจำนวนชั้นและ Neuron ได้)  ถ้ามีหลายชั้นเรียกว่า Deep Learning")
st.markdown("- Output Layer → ส่งผลลัพธ์ออกตามรูปแบบของข้อมูลที่คาดการณ์")

st.write("## 📌ขั้นตอนการพัฒนาโมเดล")
st.markdown("1. การสร้างโมเดล (Model Architecture) ฟังก์ชัน create_model() ใช้ TensorFlow/Keras สร้างโมเดลตามโครงสร้างที่กล่าวไว้")

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

st.markdown("2. แสดงโครงสร้างของโมเดล ใช้ model.summary() เพื่อแสดงรายละเอียดของโมเดล เช่น จำนวนพารามิเตอร์และ Layer ต่างๆ")

code = '''# Function to convert model summary to string
def get_model_summary(model):
    string_io = io.StringIO()
    model.summary(print_fn=lambda x: string_io.write(x + "\n"))
    summary_string = string_io.getvalue()
    string_io.close()
    return summary_string

# Display model summary
st.subheader("Model Summary")
summary_string = get_model_summary(model)

st.code(summary_string, language="text")
'''
st.code(code, language="python")

st.markdown("3. การฝึกโมเดล (Model Training)\n "
"\n- กดปุ่ม 'Train Model' เพื่อเริ่มการฝึก\n "
"\n- ใช้ model.fit() เทรนโมเดล 5 รอบ (epochs = 5)\n "
"\n- ใช้ batch_size = 32 เพื่ออัปเดตค่าถ่วงน้ำหนักทุก ๆ 32 ตัวอย่าง \n"
"\n- ใช้ validation_split = 0.1 เพื่อแบ่งข้อมูล 10% สำหรับตรวจสอบความถูกต้องของโมเดล")

code = '''if st.button("Train Model"):
    with st.spinner("Training in progress..."):
        history = model.fit(X, y, epochs=5, batch_size=32, validation_split=0.1, verbose=0)
    st.success("Training complete!")'''
st.code(code, language="python")

st.markdown("4. แสดงผลการฝึกโมเดล\n "
"\n- history.history['accuracy'] → ค่าความแม่นยำของชุด Train\n "
"\n- history.history['val_accuracy'] → ค่าความแม่นยำของชุด Validation")

code = '''# Display results
    st.write("Final Training Accuracy:", history.history['accuracy'][-1])
    st.write("Final Validation Accuracy:", history.history['val_accuracy'][-1])'''
st.code(code, language="python")

st.markdown("5.แสดงกราฟผลการฝึก (Training Visualization) ใช้ Matplotlib วาดกราฟ:\n"
"\n- Accuracy (ความแม่นยำของโมเดล)\n"
"\n- Loss (ค่าความผิดพลาด)")

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

st.markdown("<p style='text-align: right;'>อ้างอิงจาก chatGPT</p>",
            unsafe_allow_html=True)

