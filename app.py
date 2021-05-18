from tensorflow import keras
from PIL import Image
import numpy as np
from numpy import array
from skimage import transform
import streamlit as st
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
import os
import io
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)

MODELSPATH = "C:/Users/cwh93/Desktop/Final_Project/model/"
DATAPATH = "C:/Users/cwh93/Desktop/Final_Project/data/"


st.write("""        
            # Check Your Moles!
Check your weird looking moles before it gets too late.

""")

code = {'AK': 0, 'BCC': 1, 'BKL': 2, 'DF': 3, 'MEL': 4, 'NV': 5, 'SCC': 6, 'VASC': 7}

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




m_c=    array([[ 115,   35,    9,    0,    8,    0,    7,    0],
       [  41,  591,    6,    1,   16,    2,    6,    2],
       [  39,   51,  324,    1,   66,   33,   10,    1],
       [   2,   10,    0,   32,    1,    2,    1,    0],
       [  16,   47,   59,    2,  634,  139,    7,    1],
       [  16,  145,  163,    8,  300, 1916,   17,   10],
       [   9,   24,    5,    0,    5,    4,   79,    0],
       [   0,    4,    2,    0,    5,    2,    0,   38]])

true_n=[]
true_p=[]
false_p=[]
false_n=[]
for s in [2,3,5,7]:
    for d in [0,1,4,6]:
        true_n.append(m_c[s][s])
        true_p.append(m_c[d][d])
        false_p.append(m_c[d][s])
        false_n.append(m_c[s][d])
b_c=np.array([[sum(true_n)/4,sum(false_p)],[sum(false_n),sum(true_p)/4]])

def plot_cm(cm):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cbar=False, cmap='Blues')
    if len(cm)==2:

        plt.xticks(np.arange(2) + 0.5, ['not_dangerous','dangerous'])

        plt.yticks(np.arange(2) + 0.5, ['not_dangerous','dangerous'])
        plt.title("Binary Confusion Matrix")
    if len(cm)==7:

        plt.xticks(np.arange(7) + 0.5, code.keys())
        plt.yticks(np.arange(7) + 0.5, code.keys())
        plt.title("Multi-class Confusion Matrix")
    
    
    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    

    plt.show()




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


def danger(reslt):
    if reslt[0] in [0,1,4,6]:
        st.write('**Our model could find your skin lesion to be similar to one of the dangerous types**')
        st.write(lesion_type_dict[reslt[0]] ,'is one of the lesions that are dangerous or need to be checked up.')
        st.write('Please visit a dermatologist soon.')
        st.write('**please do not ignnore:**')
        st.write('If our model said it looks dangerous, the probability of it being false is:',str((sum(false_p)/(sum(true_p)/4+sum(false_p)))*1000000//100/100)+'%')
        plot_cm(b_c)
    else:
        st.write('Our model **could not** find your skin lesion to be similar to one of the dangerous types')
       
        st.write(lesion_type_dict[reslt[0]] ,'is one of the lesions that are safe!')
        st.write("congratulations. You'll live.")

        st.write('**please do not ignnore:**')
        st.write('If our model said it looks safe, the probability of it being false is:',str((sum(false_n)/(sum(true_n)/4+sum(false_n)))*1000000//100/100)+'%')   
        plot_cm(b_c)   
    if st.checkbox('Binary Confusion matrix'): 
        st.pyplot(plot_cm(b_c))
    




model = load_model(MODELSPATH + 'model_final2.h5')

def output(input_):
    st.success("We have recieved the image. Processing...")
    if st.checkbox('Proceed now',key=input_[0]):
        image = load(DATAPATH + input_[0]+'.jpg')
        reslt=np.argmax(model.predict(image), axis=1)
        danger(reslt)

        st.write('The detailed result is: '+lesion_type_dict[reslt[0]])
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
        danger(reslt)
        st.header("What does it mean?")
        st.write(definition_dict[reslt[0]])
        st.write(lesion_type_dict[reslt[0]],desc_dict[reslt[0]])







st.sidebar.title("About")
st.sidebar.info("The motivation for making of this app is to improve the accessibility so motivate people to get the check up.")
st.sidebar.info("The application classifies different skin lesion(Actinic keratosis, intraepithelial carcinoma, basal cell carcinoma, melanoma, Squamous Cell Carcinoma, benign keratosis-like lesions, melanocytic nevi, dermatofibroma, vascular lesions). This app is built using a Convolution Neural Network.")





