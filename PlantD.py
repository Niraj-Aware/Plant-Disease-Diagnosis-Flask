import streamlit as st
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = tf.keras.models.load_model('PlantDNet.h5', compile=False)
print('Model loaded.')

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255
    preds = model.predict(x)
    return preds

def main():
    st.title("Plant Disease Classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        st.write("")
        st.write("Classifying...")
        with st.spinner("Wait for it..."):
            with open("temp_image.jpg", "wb") as f:
                f.write(image)
            preds = model_predict("temp_image.jpg", model)
            ind = np.argmax(preds[0])
            disease_class = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                             'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                             'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                             'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                             'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
            prediction = disease_class[ind]
            st.write(f"Prediction: {prediction}")

if __name__ == '__main__':
    main()
