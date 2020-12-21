#UNET code adapted from https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb
from google.cloud import storage
import os
import subprocess

from platform import python_version

print(python_version())

import scipy.misc

import random
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from PIL import Image

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from skimage import img_as_uint
from imageio import imwrite
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#@title Authenticate
# from google.colab import auth
# auth.authenticate_user()
# print('Authenticated')

# Set some parameters
im_width = 512
im_height = 512
border = 5
USE_PRETRAINED = False
THIS_FOLDER = os.path.dirname(os.path.abspath("."))
MODEL_PATH = os.path.join(THIS_FOLDER, 'unet-pretrained-weights.h5')

project = "dendritic-spines"
#storage_client = storage.Client(project)
storage_client = storage.Client.from_service_account_json('/home/pkhart/dendritic-spines/train/bridge-urops-312a328069d8.json')
bucket = storage_client.get_bucket(project)

#download image indexer json. This is used to standardize training, testing, and validation splits.
ids = json.loads(bucket.get_blob('layerTraining/fileIndexList.json').download_as_string().decode('utf-8'))
#ids = json.loads("./fileIndexList.json")
print("No. of images = ", len(ids))

X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)


IMAGE_DIR = "images/"
MASK_DIR = "masks/"
DATA_PATH = "/home/pkhart/dendritic-spines/train/data/"

PNG = '.png'
cwd = os.getcwd()
print(cwd)
print(os.path.join(DATA_PATH, IMAGE_DIR))
for prefix in [IMAGE_DIR, MASK_DIR]:
    os.makedirs(os.path.join(DATA_PATH, prefix), exist_ok = True )

    # tqdm is used to display the progress bar
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
            # Load images
        img_path = prefix + id_ + PNG
        local_path = os.path.join(DATA_PATH, img_path)
        try:

            if not os.path.exists(local_path): #check if image already stored.
                with open(local_path, 'w'): #create an empty file to store the image
                    pass
                #download the image or mask from the bucket
                bucket.get_blob('jupyter/'+img_path).download_to_filename(local_path)

            #load the images into numpy arrays and normalize between 0 and 1
            if prefix == IMAGE_DIR:

                x_img = img_to_array(imread(local_path))
                X[n] = x_img/255.0
            else:
                mask = img_to_array(load_img(local_path, color_mode = "grayscale"))
                y[n] = mask/255.0
        except ValueError:
            print("can't load " + local_path + " due to value error")
#         except:
#             print("cannot identify image file " + local_path)
    #augment: transpose, rotate, flip (keras)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
num_examples = 4
fig, ax = plt.subplots(num_examples, 2, figsize = (20, 15))

for i in range(num_examples):
    ix = random.randint(0, len(X_train))
    ax[i,0].imshow(X[ix][:,:,0])
    ax[i,1].imshow(y[ix][:,:,0])
    ax[i,0].set_title(ids[ix])

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"]) #maybe try jaccard loss?


model.summary()
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1), 
    ModelCheckpoint(MODEL_PATH, verbose=1, save_best_only=True, save_weights_only=True) #increase min_lr, also look at keras hyper param tuner (batch size, learning rate, dropout)
]


if USE_PRETRAINED:
    if not os.path.exists(MODEL_PATH): #check if image already stored.
        with open(local_path, 'w'): #create an empty file to store the image
            pass
        #download the pretrained modle from the bucket
        bucket.get_blob('jupyter/unet-pretrained-weights.h5').download_to_filename(MODEL_PATH)
else:
    #model.load_weights(MODEL_PATH)
    print("TRAINING!")
    results = model.fit(X_train, y_train, batch_size=4, epochs=50, callbacks=callbacks,validation_data=(X_valid, y_valid))
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend();
    #bucket.get_blob('jupyter/unet-pretrained-weights.h5').download_to_filename(local_path)

    bucket.blob('jupyter/unet-pretrained-weights.h5').upload_from_filename(MODEL_PATH)
    
model.load_weights(MODEL_PATH)

model.evaluate(X_valid, y_valid, verbose=1)
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)

def infer_all():
    print("saving")
    all_inference = (model.predict(X, verbose=1) > 0.5).astype(np.bool)
    for i in range(len(ids)):
        array = img_as_uint(all_inference[i,:,:,0]*255)
        filename = "/home/pkhart/dendritic-spines/train/data/infrence/"+ids[i]+PNG
        imwrite(array, filename)

            
       
infer_all()


def plot_sample(X, y, preds, binary_preds, N):
    for n in range(N):
        #print(str(X.shape) + "_" + str(y.shape) + "_" + str(binary_preds.shape))

        ix = random.randint(0, len(X))
        #print(X[ix,:,:,0])
        f, axarr = plt.subplots(2,3)
        axarr[0,0].imshow(X[ix,:,:,0])
        axarr[0,0].set_title("image")
        axarr[0,1].imshow(y[ix,:,:,0])
        axarr[0,1].set_title("mask")
        axarr[0,2].imshow(binary_preds[ix,:,:,0])
        axarr[0,2].set_title("predicted")

        axarr[1,0].imshow(X[ix,:,:,0], cmap = 'gray')
        axarr[1,1].imshow(X[ix,:,:,0], cmap = 'gray')
        axarr[1,2].imshow(X[ix,:,:,0], cmap = 'gray')

        axarr[1,1].imshow(y[ix,:,:,0], cmap = 'Greens', alpha = 0.75)

        axarr[1,2].imshow(binary_preds[ix,:,:,0], cmap = 'Greens', alpha = 0.75)
        plt.show()

plot_sample(X_valid, y_valid, preds_val, preds_val_t, 10)












