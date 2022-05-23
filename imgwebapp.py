
import json
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

st.legacy_caching.caching.clear_cache()

response=False

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Potato Image Classifier")
st.text("Provide an image of Potato Leaf")

menu = ["Image","Camera"]
choice = st.sidebar.selectbox("Menu",menu)


@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('/app/models/1')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

loc_button = Button(label="Get Location",button_type="success")

loc_button.js_on_event( "button_click",  CustomJS(code="""
    navigator.geolocation.getCurrentPosition(
        (loc) => {
            document.dispatchEvent(new CustomEvent("GET_LOCATION", {detail: {lat: loc.coords.latitude, lon: loc.coords.longitude}}))
        }
    )
    """))
result = streamlit_bokeh_events(
    loc_button,
    events="GET_LOCATION",
    key="get_location",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0)
  
print()

if result:
  lat=result['GET_LOCATION']['lat']
  lon=result['GET_LOCATION']['lon']
  response = requests.get("https://api.openweathermap.org/data/2.5/weather?lat={}&lon={}7&appid=5df97ab497c61f4c567e538686631fde&units=metric".format(lat,lon))
  json_data = json.loads(response.text)


if response:
   st.write("Weather : {} ".format(json_data['weather'][0]['main']))
   st.write("Description : {} ".format(json_data['weather'][0]['description']))
   st.write("Temperature : {} Celcius ".format(int(json_data['main']['temp'])))
   st.write("Humidity : {} ".format(int(json_data['main']['humidity'])))
   st.write("Wind Speed : {} ".format(int(json_data['wind']['speed'])))
   st.write("City : {}".format(json_data['name']))
   if json_data['sys']['country'] == "IN":
      st.write("Country : {} ".format("India"))
   else:
      st.write("Country: {}  ".format(json_data['sys']['country']))

   

if choice == "Image":
    st.subheader("Image")
    file= st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    if file is not None:

        image= Image.open(file)
    
        img_array = np.array(image)
    
        img_tensor = tf.cast(img_array, tf.float32)
        img = tf.image.resize(img_tensor,[256,256])
        img = np.expand_dims(img, axis = 0)
        
        st.write("Predicted Class :")
        with st.spinner('classifying.....'):
          pred = model.predict(img)
          
          label =np.argmax(pred,axis=1)
          
          confidence = "{:.2f}".format(100*np.max(pred))
          
          st.write(classes[label[0]]) 
          
          st.write("Confidence :", confidence) 
          st.write("")
          image = Image.open(file)
          st.image(image, caption='Classifying Potato Image', width=400)
          


elif choice == "Camera":
  st.subheader("Camera")
  
  picture = st.camera_input("Take a picture",)

  if picture:
      image= Image.open(picture)
      
      img_array = np.array(image)
    
      img_tensor = tf.cast(img_array, tf.float64)
      img = tf.image.resize(img_tensor,[256,256])
      img = np.expand_dims(img, axis = 0)
        
      st.write("Predicted Class :")
      with st.spinner('classifying.....'):
        pred = model.predict(img)
          
        label =np.argmax(pred,axis=1)
          
        confidence = "{:.2f}".format(100*np.max(pred))
          
        st.write(classes[label[0]]) 
          
        st.write("Confidence :", confidence) 
        st.write("")
        image = Image.open(picture)
        st.image(image, caption='Classifying Potato Image', width=400)
        
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

