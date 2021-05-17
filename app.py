from tensorflow import keras
from PIL import Image
import numpy as np
from skimage import transform
import streamlit as st
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
import os
import io
import pandas as pd

MODELSPATH = "C:/Users/cwh93/Desktop/Final_Project/model/"
DATAPATH = "C:/Users/cwh93/Desktop/Final_Project/data/"


st.write("""

        
            # Check Your Moles!
        

Check your weird looking moles before it gets too late.

""")

@st.cache
def show_img(name):
    img = Image.open(DATAPATH + name+'.jpg')
    return img

def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

lesion_type_dict = {2: '**Benign keratosis-like lesions**', 5: '**Melanocytic nevi**', 3: '**Dermatofibroma**',
                        4: '**Melanoma**',6:'**Squamous cell carcinoma**', 7: '**Vascular lesions**', 1: '**Basal cell carcinoma**', 0: '**Actinic keratosis**'}

definition_dict={0:'pre-cancerous, needs to be checked out',1:'cancer that in the stratum basal of the epidermis. (lowest part of the skin above the dermis)', 
                        2:'non-cancerous, not dangerous', 3:'not dangerous',5:'moles, birthmarks-not dangerous',
                        4:'cancer of the melanocytes, Most Dangerous',6:'dangerous',7:'not dangerous'}

desc_dict={0:"is a rough, scaly patch on the skin that develops from years of sun exposure. It's often found on the face, lips, ears, forearms, scalp, neck or back of the hands. It's the most common precancer.(10% will become cancerous)",
            1:"is the most common form of skin cancer and the most frequently occurring form of all cancers. The prognosis for patients with BCC is excellent, with a 100% survival rate for cases that have not spread to other sites. ",
            2:"is a common noncancerous skin growth. People tend to get more of them as they get older. Benign keratosis-like lesions (Seborrheic keratoses) are usually brown, black or light tan. The growths look waxy, scaly and slightly raised.",
            3:"is the overlying epidermis is slightly thickened. Their occurrence is not unusual in children and adolescents. Dermatofibromas are firm and may be black, red, brown, or flesh-colored. Their diameter generally ranges from 0.5 to l.5 cm, although they may occasionally be larger. Dermatofibromas may be solitary or multiple, and they develop either spontaneously or after minor trauma to the skin, such as an insect bite.",
            5:"(also known as nevocytic nevus, nevus-cell nevus and commonly as a mole) is a type of melanocytic tumor that contains nevus cells. Some sources equate the term mole with melanocytic nevus, but there are also sources that equate the term mole with any nevus form.",
            4:"is a form of skin cancer that begins in the cells (melanocytes) that control the pigment in your skin. This illustration shows melanoma cells extending from the surface of the skin into the deeper skin layers.",
            6:"is usually not life-threatening, though it can be aggressive. Untreated, squamous cell carcinoma of the skin can grow large or spread to other parts of your body, causing serious complications.",
            7:"is relatively common abnormalitie of the skin and underlying tissues, more commonly known as birthmarks."
            }

model = load_model(MODELSPATH + 'model_final2')

def output(input_):
    st.success("We have recieved the image. Processing...")
    if st.checkbox('Proceed now',key=input_[0]):
        image = load(DATAPATH + input_[0]+'.jpg')
        reslt=np.argmax(model.predict(image), axis=1)
        st.write('The result is: '+lesion_type_dict[reslt[0]])
        if reslt[0] in [0,1,5,6]:
            st.write(lesion_type_dict[reslt[0]] ,'is one of the lesions that are dangerous or need to be checked up.')
            st.write('Please visit a dermatologist soon.')
        st.header("What does it mean?")
        st.write(definition_dict[reslt[0]])
        st.write(lesion_type_dict[reslt[0]],desc_dict[reslt[0]])


    st.info("This is the image you selected")
    st.image(show_img(input_[0]), caption='Sample Data', use_column_width=True)


def output_2(input_):
    st.success("We have recieved the image. Processing...")
    model = load_model(MODELSPATH + 'model.h5')
    if st.checkbox('Proceed now'):
        image = load(DATAPATH + input_+'.jpg')






st.sidebar.header('Skin Cancer Detector')
st.sidebar.subheader('Try it with:')

try_ = st.sidebar.selectbox("", ["Skin lesions sample images", "Pictures of your skin lesions"])
if try_ == "Skin lesions sample images":
    st.header("Skin lesions sample images")
    st.markdown("""
        **We are running the model with picutes that are properly taken with medical scope.**

        This is the best example of the data we submit.

        Your choice.
        """)
    

    Skin_leision_example = ['AK','BCC','MEL','SCC','BKL','NV','DF','VASC']
    ex_chosen = st.multiselect('Sample Leision', Skin_leision_example)

    if len(ex_chosen) > 1:
        
        for q in ex_chosen:
            l=[]
            l.append(q)
            output(l)




    if len(ex_chosen) == 1:
        output(ex_chosen)
        

        
    else:
        st.info('Please select Sample Leision')
if try_=="Pictures of your skin lesions":
    st.header("Please upload your image(224X224 or higher resolution.)")
    st.write("Remember: For the best accuracy, please submit the picture that are taken about within 3 inches far, the lesion to be centered alone.")

    file_path = st.file_uploader('Upload an image', type=['png', 'jpg'])
    if file_path is not None:
        image = load(file_path)


        reslt=np.argmax(model.predict(image), axis=1)
        st.write('The result is: '+lesion_type_dict[reslt[0]])
        if reslt[0] in [0,1,5,6]:
            st.write(lesion_type_dict[reslt[0]] ,'is one of the lesions that are dangerous or need to be checked up.')
            st.write('Please visit a dermatologist soon.')
        st.header("What does it mean?")
        st.write(definition_dict[reslt[0]])
        st.write(lesion_type_dict[reslt[0]],desc_dict[reslt[0]])





st.sidebar.title("About")
st.sidebar.info("The motivation for making of this app is to improve the accessibility so motivate people to get the check up.")
st.sidebar.info("The application classifies different skin lesion(Actinic keratosis, intraepithelial carcinoma, basal cell carcinoma, melanoma, Squamous Cell Carcinoma, benign keratosis-like lesions, melanocytic nevi, dermatofibroma, vascular lesions). This app is built using a Convolution Neural Network.")





