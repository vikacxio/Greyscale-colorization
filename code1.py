import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization,LeakyReLU
from keras.layers.merge import concatenate
from keras.models import Model
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

tf.random.set_seed(123)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
tf.random.set_seed(2)
np.random.seed(1)

print(os.listdir("C:/Users/vikac/Downloads/Newfolder2/input/dataset/dataset_updated"))

ImagePath="C:/Users/vikac/Downloads/Newfolder2/input/dataset/dataset_updated/training_set/painting/"
img = cv2.imread(ImagePath+"0205.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
plt.imshow(img)
img.shape

HEIGHT = 224
WIDTH = 224
ImagePath = "C:/Users/vikac/Downloads/Newfolder2/input/dataset/dataset_updated/training_set/painting/"


def ExtractInput(path):
    X_img = []
    y_img = []
    for imageDir in os.listdir(ImagePath):
        try:
            img = cv2.imread(ImagePath + imageDir)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            img = img.astype(np.float32)
            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
            # Convert the rgb values of the input image to the range of 0 to 1
            # 1.0/255 indicates that we are using a 24-bit RGB color space.
            # It means that we are using numbers between 0–255 for each color channel
            # img_lab = 1.0/225*img_lab
            # resize the lightness channel to network input size
            img_lab_rs = cv2.resize(img_lab, (WIDTH, HEIGHT))  # resize image to network input size
            img_l = img_lab_rs[:, :, 0]  # pull out L channel
            # img_l -= 50 # subtract 50 for mean-centering
            img_ab = img_lab_rs[:, :, 1:]  # Extracting the ab channel
            img_ab = img_ab / 128
            # The true color values range between -128 and 128. This is the default interval
            # in the Lab color space. By dividing them by 128, they too fall within the -1 to 1 interval.
            X_img.append(img_l)
            y_img.append(img_ab)
        except:
            pass
    X_img = np.array(X_img)
    y_img = np.array(y_img)

    return X_img, y_img



X_,y_ = ExtractInput(ImagePath) # Data-preprocessing

K.clear_session()


def InstantiateModel(in_):
    model_ = Conv2D(16, (3, 3), padding='same', strides=1)(in_)
    model_ = LeakyReLU()(model_)
    # model_ = Conv2D(64,(3,3), activation='relu',strides=1)(model_)
    model_ = Conv2D(32, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)
    model_ = MaxPooling2D(pool_size=(2, 2), padding='same')(model_)

    model_ = Conv2D(64, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)
    model_ = MaxPooling2D(pool_size=(2, 2), padding='same')(model_)

    model_ = Conv2D(128, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)

    model_ = Conv2D(256, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)

    model_ = UpSampling2D((2, 2))(model_)
    model_ = Conv2D(128, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)

    model_ = UpSampling2D((2, 2))(model_)
    model_ = Conv2D(64, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    # model_ = BatchNormalization()(model_)

    concat_ = concatenate([model_, in_])

    model_ = Conv2D(64, (3, 3), padding='same', strides=1)(concat_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)

    model_ = Conv2D(32, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    # model_ = BatchNormalization()(model_)

    model_ = Conv2D(2, (3, 3), activation='tanh', padding='same', strides=1)(model_)

    return model_


Input_Sample = Input(shape=(HEIGHT, WIDTH,1))
Output_ = InstantiateModel(Input_Sample)
Model_Colourization = Model(inputs=Input_Sample, outputs=Output_)


LEARNING_RATE = 0.001
Model_Colourization.compile(optimizer=Adam(lr=LEARNING_RATE),
                            loss='mean_squared_error')
Model_Colourization.summary()


def GenerateInputs(X_,y_):
    for i in range(len(X_)):
        X_input = X_[i].reshape(1,224,224,1)
        y_input = y_[i].reshape(1,224,224,2)
        yield (X_input,y_input)
Model_Colourization.fit_generator(GenerateInputs(X_,y_),epochs=10,verbose=1,steps_per_epoch=10,shuffle=True)#,validation_data=GenerateInputs(X_val, y_val))

TestImagePath="C:/Users/vikac/Downloads/Newfolder2/input/dataset/dataset_updated/training_set/iconography/"


def ExtractTestInput(ImagePath):
    img = cv2.imread(ImagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_ = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2Lab)
    img_ = img_.astype(np.float32)
    img_lab_rs = cv2.resize(img_, (WIDTH, HEIGHT))  # resize image to network input size
    img_l = img_lab_rs[:, :, 0]  # pull out L channel
    # img_l -= 50
    img_l_reshaped = img_l.reshape(1, 224, 224, 1)

    return img_l_reshaped


ImagePath=TestImagePath+"0bzkK4.jpg"
image_for_test = ExtractTestInput(ImagePath)
Prediction = Model_Colourization.predict(image_for_test)
Prediction = Prediction*128
Prediction=Prediction.reshape(224,224,2)


plt.figure(figsize=(30,20))
plt.subplot(5,5,1)
img = cv2.imread(TestImagePath+"0bzkK4.jpg")
img_1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img = cv2.resize(img, (224, 224))
plt.imshow(img)

plt.subplot(5,5,1+1)
img_ = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
img_[:,:,1:] = Prediction
img_ = cv2.cvtColor(img_, cv2.COLOR_Lab2RGB)
plt.title("Predicted Image")
plt.imshow(img_)

plt.subplot(5,5,1+2)
plt.title("Ground truth")
plt.imshow(img_1)

