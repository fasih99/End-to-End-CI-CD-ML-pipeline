import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Potato Leaf Disease Classifier")
st.text("Provide URL of potato leaf Image for image classification")

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('/app/models/1/')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes=['Potato__Early_blight','Potato__healthy','Potato__Late_blight']

def decode_img(image):
  img = tf.image.decode_jpeg(image, channels=3)  
  img = tf.image.resize(img,[256,256])
  return np.expand_dims(img, axis=0)

path = st.text_input('Enter Image URL to Classify.. ','https://img2.lrgarden.com/feed_pic/122/13/1000333946_1000013406_1505279245.jpg')
if path is not None:
    content = requests.get(path).content

    st.write("Predicted Class :")
    with st.spinner('classifying.....'):
      label =np.argmax(model.predict(decode_img(content)),axis=1)
      st.write(classes[label[0]])    
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying Potato Leaf Diseases', use_column_width=True)

