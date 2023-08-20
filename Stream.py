import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf

# Load the pre-trained model
model = load_model('PlantDNet.h5')  # Update with the correct model path
disease_class = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

def classify_image(img,model):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255

    with tf.autograph.experimental.do_not_convert():
        custom = model.predict(x)
    ind = np.argmax(custom[0])
    return disease_class[ind]

def main():
    st.title("Plant Disease Classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(image, caption="Uploaded Image.", use_column_width=True)

        st.write("")
        st.write("Classifying...")
        prediction = classify_image(image)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
