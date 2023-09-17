import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
#import cv2

fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Retina Image Classification Group 30')

st.markdown("Welcome to web application that classifies Diabetic Retinopathy that have been separated into 5 categories :  No apparent retinopathy, Mild Non-proliferative , Moderate Non-proliferative , Severe Non-proliferative, Proliferative")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)


def predict(image):
    classifier_model = "retina_best_seq.h5"
    IMAGE_SHAPE = (28, 28,3)
    model = load_model(classifier_model,compile=False,)
    
    data = np.ndarray(shape=(1,28,28,3))
    test_image = image
    size=(28,28)
    test_image = ImageOps.fit(test_image,size,Image.ANTIALIAS)
    
    image_array = np.asarray(test_image)
    
    normalized_image_array = image_array /255.0
    
    data[0]= normalized_image_array
    
    #test_image = image.resize((28,28),Image.ANTIALIAS)
    #test_image = preprocessing.image.img_to_array(test_image)
    #test_image = test_image / 255.0
    #test_image = np.expand_dims(test_image, axis=0)
    
    
    #test_image = np.repeat(test_image,-1, axis = 0)
    class_names = ['No retinopathy: class 0',
          'Mild retinopathy: class 1',
          'Moderate retinopathy: class 2',
          'Severe retinopathy: class 3',
          'Proliferative: class 4 '
          ]
    predictions = model.predict(data)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
          'No': 0,
    'Mild': 0,
    'Moderate': 0,
    'Severe': 0,
    'Proliferative': 0
    
    }


    
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result




  

if __name__ == "__main__":
    main()
