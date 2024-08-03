import streamlit as st
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from tensorflow.keras import layers,Sequential,callbacks
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


@st.cache_resource
def load_model():
    feature_extractor_layer = MobileNetV2(weights="imagenet", include_top=False,
                                     input_tensor=Input(shape=(160,160,3)))
    return feature_extractor_layer


if 'images1' not in st.session_state:
    st.session_state.images1=[]
if 'images2' not in st.session_state:
    st.session_state.images2=[]
if 'class1_name' not in st.session_state:
    st.session_state.class1_name=''
if 'class2_name' not in st.session_state:
    st.session_state.class2_name=''

class CustomCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
       
        # my_bar.progress(epoch)
        keys = list(logs.keys())
        st.session_state.my_bar.progress((epoch+1)*2,'Model Training')
        print('myprogress is',st.session_state.progress)
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

def train():
    data = []
    labels = []
    for imagePath in st.session_state.images1:
        # extract the class label from the filename
        # label = imagePath.split(os.path.sep)[-2]
        # load the input image (224x224) and preprocess it
        image = load_img(imagePath, target_size=(160, 160))
        image = img_to_array(image)
        image = image/255
        #image = preprocess_input(image)

            # update the data and labels lists, respectively
        data.append(image)
        labels.append(0)
    for imagePath in st.session_state.images2:
        # extract the class label from the filename
        # label = imagePath.split(os.path.sep)[-2]
        # load the input image (224x224) and preprocess it
        image = load_img(imagePath, target_size=(160, 160))
        image = img_to_array(image)
        image = image/255
        #image = preprocess_input(image)

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(1)

    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    print(data.shape)


    feature_extractor_layer = load_model()
    feature_extractor_layer.trainable = False
    model = Sequential()
    model.add(feature_extractor_layer)
    model.add(Flatten(name="flatten"))
    model.add(Dense(1024, activation='relu', name='hidden_layer'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', name='output'))

    model.compile(optimizer=Adam(learning_rate=1e-5),loss="categorical_crossentropy",metrics=["accuracy"])


    aug = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")
    labels = to_categorical(labels)

    model.fit(aug.flow(data, labels),epochs=50,callbacks=[CustomCallback()])
    return model