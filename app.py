from tensorflow import keras
from PIL import Image
import numpy as np
from skimage import transform
import streamlit as st


st.writ("""
#Check Your Moles!

Check your weird looking moles before it gets too late!

""")

st.sidebar.title("About")
st.sidebar.info("The motivation for making of this app is to improve the accessibility so motivate people to get the check up. /nThe application classifies different skin lesion(Actinic keratosis, intraepithelial carcinoma, basal cell carcinoma, melanoma, Squamous Cell Carcinoma, benign keratosis-like lesions, melanocytic nevi, dermatofibroma, vascular lesions). /nIt was built using a Convolution Neural Network.")





new_model = keras.models.load_model('/content/12.h5')

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

image = load('/content/drive/MyDrive/Dataset2/classes_test/MEL/ISIC_0001105_downsampled.jpg')
np.argmax(new_model.predict(image), axis=1)