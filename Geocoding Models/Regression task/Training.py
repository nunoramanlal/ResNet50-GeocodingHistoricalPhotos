import keras
from PIL import Image
from io import BytesIO
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras import optimizers, backend as K
from sklearn.cross_validation import train_test_split
from skimage.color import rgb2lab
import sys
from data_utils import geodistance_theano, normalize_values
from model_utils_new import regression_model
from keras.callbacks import ModelCheckpoint
import keras

import tensorflow as tf
import numpy as np
import random as rn
import os

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.DeviceCountEntry
session_conf.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(graph=tf.get_default_graph(),config=session_conf))

_BOUNDING_BOX = [37.639830, 37.929824, -123.173825, -122.281780] #sf
#_BOUNDING_BOX = [40.477399,  40.917577, -74.259090, -73.700272] #ny
_MODEL_FINAL_NAME = 'mobilenet_sf_regression.h5'
_MODEL_WEIGHTS_FINAL_NAME = 'mobilenet_sf_regression_weights.h5'
#_PATH = '../../../../home/nramanlal/data-nramanlal-lfreixinho/resized_images_sf/'
_PATH = '../resized_images_sf2/'
_X_TEST_FILE = 'x_test_regression_sf.txt'
_TRAIN_TEST_SPLIT_SEED = 2
num_per_epoch = 100
batch_size = 32

def get_images(path):
    images_list = os.listdir(path) #list of all images
    images = []
    coordinates = []
    for line in images_list:
        images.append(line)
        entry = os.path.splitext(line)[0].split(",") #filename without the extension
        coordinates.append((entry[2].rstrip(), entry[1]))
    return images, coordinates

def generate_arrays_from_file(X, Y, batch_size):
    while 1:
        line = -1
        new_X = np.zeros((batch_size, 224, 224, 3))
        new_Y = np.zeros((batch_size, 2))
        count = 0
        for entry in X:
            if count < batch_size:
                line+=1
                x_b = load_img(_PATH+entry)
                x_b = img_to_array(x_b)
                x_b = np.expand_dims(x_b, axis=0)
                x_b = x_b/255 #conversion to a range of -1 to 1. Explanation saved.
                x_b = rgb2lab(x_b)

                a = normalize_values(Y[line][0], _BOUNDING_BOX[0], _BOUNDING_BOX[1] )
                b = normalize_values(Y[line][1],_BOUNDING_BOX[2], _BOUNDING_BOX[3])
                y = [(float(a), float(b))]

                new_X[count,:] = x_b
                new_Y[count,:] = np.array(y)
                count+=1
            else:
                yield (new_X, new_Y)
                count = 0
                new_X = np.zeros((batch_size, 224, 224, 3))
                new_Y = np.zeros((batch_size, 2))

        if(np.count_nonzero(new_X) != 0):
                yield (new_X, new_Y)

#Load training images
training_images, training_coordinates = get_images(_PATH)
X_train, X_test, Y_train, Y_test = train_test_split(training_images, training_coordinates,
                                    test_size=0.20, random_state = _TRAIN_TEST_SPLIT_SEED)
X_train_size = len(X_train)

toWrite = ''
for inst in X_test:
    toWrite += inst
    toWrite += '\n'

file = open(_X_TEST_FILE, 'w')
file.write(str(toWrite))
file.close()

print('IMAGES LOADED')

model = regression_model()
checkpoint = ModelCheckpoint(_MODEL_WEIGHTS_FINAL_NAME, monitor='loss', verbose=1,
                             save_best_only=False, save_weights_only=True)

opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

model.compile(loss=geodistance_theano, optimizer = opt)
earlyStopping=keras.callbacks.EarlyStopping(monitor = 'loss', patience=1)
history = model.fit_generator(generate_arrays_from_file(X_train, Y_train, batch_size),
                            epochs=num_per_epoch,
                            steps_per_epoch=X_train_size/batch_size,
                            callbacks=[earlyStopping, checkpoint])

model.save(_MODEL_FINAL_NAME)
model.save_weights(_MODEL_WEIGHTS_FINAL_NAME)
