import streamlit as st
from streamlit_option_menu import option_menu
import train_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np



if 'images1' not in st.session_state:
    st.session_state.images1=[]
if 'images2' not in st.session_state:
    st.session_state.images2=[]
if 'class1_name' not in st.session_state:
    st.session_state.class1_name=''
if 'class2_name' not in st.session_state:
    st.session_state.class2_name=''
if 'new_image' not in st.session_state:
    st.session_state.new_image=''
if 'model' not in st.session_state:
    st.session_state.model=''
st.set_page_config(
        page_title='instaML',
        page_icon="🤖",
        layout="wide"
    )

selected = option_menu(None, ["Home", "Data", "Train", 'Export'], 
    icons=['house', 'cloud-upload', "gear", 'rocket'], 
    menu_icon="cast", default_index=0, orientation="horizontal")


############# Home Tab

if selected == "Home":
    st.title('Create your first ML project')
    st.logo('Untitled.png')
    st.subheader('How to Use the App')
    st.write("Welcome to our easy-to-use machine learning application builder! Follow these simple steps to create and download your own machine learning model.")

    st.subheader('1. Data Tab')
                
    st.write("""**Upload Your Data:** Start by uploading your dataset. Make sure your data is in a compatible format (e.g., jpg,png).

**Preview Data:** After uploading, you can preview your data to ensure it's loaded correctly.""")

    st.subheader('2. Train Tab')

    st.write("""**Train Your Model:** In this tab, you can click the 'Train' button to train your machine learning model.

**Test Your Model:** After training, you can test the model by uploading new data to see its performance. 
    """)
    st.subheader('3. Export Tab')

    st.write("""**Download the Trained Model:** Once satisfied with your trained model, you can download it for future use.

**Get the Code:** Additionally, you can download a Python code snippet that shows how to load and use the trained model in your projects.""")

    st.subheader("""**Enjoy creating powerful machine learning models with just a few clicks!**""")



########## Data Tab

if selected == "Data":
    col1,col2 = st.columns([1,1])
    
    cont1 = col1.container(border=True)
    
    class1_name = cont1.text_input("Class name",key='class1',value=st.session_state.class1_name)
    if class1_name: 
        st.session_state.class1_name = class1_name
    class_title1 = class1_name if class1_name else ""
    images1 = cont1.file_uploader(f"Upload {class_title1} images",accept_multiple_files=True,type=['png','jpg','jpeg'],key='files1')
    if images1: st.session_state.images1 = images1
    if st.session_state.images1: 
        cont1.write(f'{len(st.session_state.images1)} images uploaded')
        cont1.image(st.session_state.images1,width=100) 
    
    
    cont2 = col2.container(border=True)
    
    class2_name = cont2.text_input("Class name",key='class2',value=st.session_state.class2_name)
    if class2_name: 
        st.session_state.class2_name = class2_name
    images2 = cont2.file_uploader(f"Upload {st.session_state.class2_name} images",accept_multiple_files=True,type=['png','jpg','jpeg'],key='files2')
    if images2: st.session_state.images2 = images2
    if st.session_state.images2: 
        cont2.write(f'{len(st.session_state.images2)} images uploaded')
        cont2.image(st.session_state.images2,width=100) 
        # print(type(st.session_state.images2[0]))
        # print(len(st.session_state.images2))


############### Train Tab

if selected == 'Train':
    if st.button('Train Model'):
        st.session_state.progress=0
        st.session_state.my_bar = st.progress(0,"Processing training data")
        st.session_state.model = train_model.train()
    if st.session_state.model:
        st.success('Model Trained')

    
        col3,col4 = st.columns([1,1])
        cont3 = col3.container(border=True)
        cont4 = col4.container(border=True)
        
        st.session_state.new_image = cont3.file_uploader('Upload an image to test',accept_multiple_files=False)
        
        if st.session_state.new_image:
                cont3.image(st.session_state.new_image)
                image = load_img(st.session_state.new_image, target_size=(160, 160))
                image = img_to_array(image)
                image = image/255
                pred = st.session_state.model.predict(image.reshape(1,160,160,3))
                print(np.argmax(pred))
                if np.argmax(pred)==0:
                    cont4.subheader(st.session_state.class1_name)
                    cont4.progress(round(float(pred[0][0]),2),f'Confidence Score: {round(float(pred[0][0])*100,2)}%')
                if np.argmax(pred)==1:
                    cont4.subheader(st.session_state.class2_name)
                    cont4.progress(round(float(pred[0][1]),2),f'Confidence Score: {round(float(pred[0][1])*100,2)}%')



################ Export Tab

if selected == "Export":
    if st.session_state.model:
        st.session_state.model.save('model.keras')
        # zipfile.ZipFile('model.zip', mode='w').write("model.keras")
        with open('model.keras','rb') as f:
            # st.download_button('Download Model',f,'model.zip',mime="application/zip")
            st.download_button('Download Model',f,'model.keras')
            
    st.write('Code snippet to use your model')
    code = f"""from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import numpy as np

#input image size for the ML model
IMG_SIZE = 160
CHANNELS = 3

#load the trained model and an input image
new_model = load_model('MODEL FILE PATH/model.keras')
image = load_img('IMAGE FILE PATH/IMAGE.JPG',target_size=(IMG_SIZE, IMG_SIZE))

#preprocessing
image = img_to_array(image)
image = image/255

#prediction
prediction = new_model.predict(image.reshape(1,IMG_SIZE,IMG_SIZE,CHANNELS))
index = np.argmax(prediction,axis=1)
class_name='{st.session_state.class1_name}' if index==0 else '{st.session_state.class2_name}'
confidence_score = prediction[0][index]

#print the result and confidence score
print('Class:',class_name)
print('Confifdence Score:',confidence_score)"""

    st.code(code)
