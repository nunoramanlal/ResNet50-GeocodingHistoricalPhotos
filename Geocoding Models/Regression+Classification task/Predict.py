import keras
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from skimage.color import rgb2lab
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from keras import backend as K
import tensorflow as tf
import os, sys
import glob
import _healpy, math, healpy
import pickle
from data_utils import geodistance_theano
import random as rn

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.DeviceCountEntry
session_conf.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(graph=tf.get_default_graph(),config=session_conf))

#_IMAGE_PATH = '../../../../home/nramanlal/data-nramanlal-lfreixinho/old_resized_images_sf/'
_IMAGE_PATH = '../resized_images_sf2/'
_EVAL_PATH = 'x_test_sf_regclass.txt'
_MODEL_FINAL_NAME = 'mobilenet_sf_regclass.h5'
_NAME_FINAL_FILE = 'predictions_regclass_sf.txt'

def get_images_names(numbers):
    images = []
    data = open(numbers)
    for entry in data:
        images.append(entry.rstrip())
    return images
'''
def get_images_names(path):
    images = os.listdir(path)
    final = []
    for image in images:
        final.append(image.rstrip())
    return images
'''
def get_images (images_names):
    size = len(images_names)
    while 1:
        for name in range(size):
            x_b = load_img(_IMAGE_PATH+images_names[name])
            x_b = img_to_array(x_b)
            x_b = np.expand_dims(x_b, axis=0)
            x_b = x_b/255 #conversion to a range of -1 to 1. Explanation saved.
            x_b = rgb2lab(x_b)
            yield(x_b)

images_names = get_images_names(_EVAL_PATH)

with CustomObjectScope({'geodistance_theano': geodistance_theano}):
    model = load_model(_MODEL_FINAL_NAME)

predictions = model.predict_generator(get_images(images_names), steps=len(images_names), verbose = 1)

toWrite = ''
for prediction in range(len(predictions[0])):
    toWrite += str(images_names[prediction]) + ','
    toWrite += str(predictions[0][prediction][0]) + ','+ str(predictions[0][prediction][1])
    toWrite += '\n'


file = open(_NAME_FINAL_FILE,'w')
file.write(toWrite)
file.close()
