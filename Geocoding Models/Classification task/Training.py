from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
import _healpy, math, healpy, pickle, keras, sys
from keras.utils import np_utils
from keras.preprocessing.image import load_img, img_to_array
from skimage.color import rgb2lab
from model_utils_new import classification_model
import tensorflow as tf
import numpy as np
import random as rn
import os
from keras.callbacks import ModelCheckpoint
import keras

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.DeviceCountEntry
session_conf.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(graph=tf.get_default_graph(),config=session_conf))

_MODEL_FINAL_NAME = 'mobilenet_sf_classification_9k.h5'
_MODEL_WEIGHTS_FINAL_NAME = 'mobilenet_sf_classification_weights_9k.h5'
_PATH = '../../../../home/nramanlal/data-nramanlal-lfreixinho/resized_images_sf/'
_ENCODER = 'enconder_classification_sf.p'
_X_TEST_FILE = 'x_test_classification_sf.txt'
_TRAIN_TEST_SPLIT_SEED = 2
num_epochs = 100
batch_size = 32

def get_images(path):
    images_list = os.listdir(path) #list of all images
    images = []
    coordinates = []
    for line in images_list:
        images.append(line)
        entry = os.path.splitext(line)[0].split(",") #filename without the extension
        coordinates.append((float(entry[2].rstrip()), float(entry[1]))) #(latitude, longitude)
    return images, coordinates

def generate_arrays_from_file(X, Y, batchsize):
    while 1:
        line = -1
        new_X = np.zeros((batchsize, 224, 224, 3))
        new_Y = np.zeros((batchsize, _NUMCLASSES))
        count = 0
        for entry in X:
            if count < batchsize:
                line+=1
                x_b = load_img(_PATH +entry)
                x_b = img_to_array(x_b)
                x_b = np.expand_dims(x_b, axis=0)
                x_b = x_b/255 #conversion to a range of -1 to 1. Explanation saved.
                x_b = rgb2lab(x_b)
                y= [Y[line]]

                new_X[count,:] = x_b
                new_Y[count,:] = np.array(y)
                count+=1
            else:
                yield (new_X, new_Y)
                count = 0
                new_X = np.zeros((batchsize, 224, 224, 3))
                new_Y = np.zeros((batchsize, _NUMCLASSES))

        if(np.count_nonzero(new_X) != 0):
                yield (new_X, new_Y)

training_images, training_coordinates = get_images(_PATH)
X_train, X_test, Y_train, Y_test = train_test_split(training_images, training_coordinates, test_size=0.20, random_state = _TRAIN_TEST_SPLIT_SEED)
X_train_size = len(X_train)

toWrite = ''
for inst in X_test:
    toWrite += inst
    toWrite += '\n'

file = open(_X_TEST_FILE, 'w')
file.write(str(toWrite))
file.close()

print("WRITTING DONE")

encoder = LabelEncoder()
Y_train = [ _healpy.latlon2healpix( i[0] , i[1] , math.pow(4 , 6) ) for i in Y_train ]
Y_test = [ _healpy.latlon2healpix( i[0] , i[1] , math.pow(4 , 6) ) for i in Y_test ]
fit_trans = encoder.fit_transform( Y_train + Y_test )
_encoder = np_utils.to_categorical(fit_trans)
_newenconder = _encoder.astype(int)

_NUMCLASSES = len(_newenconder[0])
print('NUM OF CLASSES --->', _NUMCLASSES )

Y_train = _newenconder[:-len(Y_test)]
Y_test = _newenconder[-len(Y_test):]

pickle.dump(encoder, open(_ENCODER, 'wb'))

model = classification_model(_NUMCLASSES)


opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

model.compile(loss='categorical_crossentropy', optimizer = opt)
earlyStopping=keras.callbacks.EarlyStopping(monitor = 'loss', patience=1)

checkpoint = ModelCheckpoint(_MODEL_WEIGHTS_FINAL_NAME, monitor='loss',
                             save_best_only=False, save_weights_only=True)

history = model.fit_generator(generate_arrays_from_file(X_train, Y_train, batch_size),
                          epochs=num_epochs,
                          steps_per_epoch=X_train_size/batch_size,
                          verbose=1,
                          callbacks=[earlyStopping, checkpoint])

model.save(_MODEL_FINAL_NAME)
model.save_weights(_MODEL_WEIGHTS_FINAL_NAME)
