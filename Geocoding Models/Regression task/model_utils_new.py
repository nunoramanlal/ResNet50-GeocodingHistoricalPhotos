from keras.applications import resnet50
from keras import models, layers, optimizers

def classification_model(number_classes):
    #Load the MobileNet model
    image_size=224
    resnet = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3), pooling = 'avg')

    #build model
    inp = layers.Input(shape=(image_size, image_size, 3))
    x = resnet (inp)
    z = layers.Dense(number_classes, activation = 'softmax')(x)
    model = models.Model(inp, z)

    return model

def regression_model():
    #Load the MobileNet model
    image_size=224
    resnet = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3), pooling = 'avg')

    #build model
    inp = layers.Input(shape=(image_size, image_size, 3))
    x = resnet (inp)
    z = layers.Dense(2, activation = 'sigmoid')(x)
    model = models.Model(inp, z)

    return model

def regre_class_model(number_classes):
    #Load the MobileNet model
    image_size=224
    resnet = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3), pooling = 'avg')

    #build model
    inp = layers.Input(shape=(224, 224, 3))
    x = resnet (inp)
    z = layers.Dense(2, activation = 'sigmoid')(x)
    q = layers.Dense(number_classes, activation = 'softmax')(x)
    model = models.Model(inp, [z,q])

    return model
