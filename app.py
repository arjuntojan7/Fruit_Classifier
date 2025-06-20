import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ✅ Load model
model = load_model("model/fruit_model.keras")

# Class names must match training order
class_names = ['apple', 'banana', 'orange']
emoji_dict = {'apple': '🍎', 'banana': '🍌', 'orange': '🍊'}

# 🎨 Page setup
st.set_page_config(page_title="Fruit Classifier", layout="centered", page_icon="🍇")

# 🎯 Header
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🍇 Fruit Image Classifier</h1>
    <p style='text-align: center; font-size: 18px;'>Upload an image of an <strong>apple</strong> , <strong>banana</strong> , or <strong>orange</strong>  to identify it.</p>
""", unsafe_allow_html=True)

# 📤 Upload image
uploaded_file = st.file_uploader("📷 Choose a fruit image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Uploaded Image", use_column_width=True)

    # 🔄 Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # 🔍 Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    # ✅ Display result
    st.success(f"### ✅ It's a **{predicted_class.capitalize()}** {emoji_dict[predicted_class]}")
    st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")

    # 📊 Probability chart
    st.subheader("🔍 Class Probabilities:")
    fig, ax = plt.subplots()
    bars = ax.bar(class_names, predictions[0], color=['#ff9999','#ffe066','#a3d977'])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probability")
    ax.set_xlabel("Fruit Class")
    ax.set_title("Prediction Confidence")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval*100:.1f}%", ha='center', va='bottom')
    st.pyplot(fig)
else:
    st.info("👈 Upload a fruit image to get started!")
