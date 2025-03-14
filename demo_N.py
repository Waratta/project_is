import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt

st.title("Demo Neural Network for mnist")

# Load dataset
file_path = "mnist_test.csv"
df = pd.read_csv(file_path)

# Split features and labels
y = df["label"].values
X = df.drop("label", axis=1).values / 255.0  # Normalize pixel values

# Define model
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
    return model

model = create_model()

# Function to convert model summary to string
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

if st.button("Train Model"):
    with st.spinner("Training in progress..."):
        history = model.fit(X, y, epochs=5, batch_size=32, validation_split=0.1, verbose=0)
    st.success("Training complete!")
    
    # Display results
    st.write("Final Training Accuracy:", history.history['accuracy'][-1])
    st.write("Final Validation Accuracy:", history.history['val_accuracy'][-1])
    
    # Plot training history
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
    
    st.pyplot(fig)