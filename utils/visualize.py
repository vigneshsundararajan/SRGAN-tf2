####################################################################################################################################

# Importing the general libraries
import os 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time

# Image Processing using Pillow
from PIL import Image

# Keras Layers, Model, and VGG imports
from tensorflow.python.data.experimental import AUTOTUNE
from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.vgg19 import VGG19

# Keras optimizers, losses, and metrics imports
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

####################################################################################################################################

# Resolving the images for accurate feature representation
# during visualization
def Resolve(model, lowres_batch):
    lowres_batch = tf.cast(lowres_batch, tf.float32)

    superres_batch = model(lowres_batch)
    superres_batch = tf.clip_by_value(superres_batch, 0, 255)
    superres_batch = tf.round(superres_batch)
    superres_batch = tf.cast(superres_batch, tf.uint8)

    return superres_batch

def ResolveSingleImage(model, lowres):
    resolved_image = Resolve(model, tf.expand_dims(lowres, axis = 0))[0]

    return resolved_image


# Loading the Images from a specified path
def LoadImage(path):
    loaded_image = np.array(Image.open(path))

    return loaded_image

####################################################################################################################################