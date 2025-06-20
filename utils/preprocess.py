from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_preprocess_image(uploaded_file, target_size=(100, 100)):
    img = image.load_img(uploaded_file, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
