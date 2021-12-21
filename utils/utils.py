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


def CropImage(lowres_image, highres_image, highres_crop_size = 96, scale = 2):
    lowres_crop_size = highres_crop_size // scale
    lowres_image_shape = tf.shape(lowres_image)[:2]

    lowres_width = tf.random.uniform(shape = (),
                             maxval = lowres_image_shape[1] - lowres_crop_size + 1,
                             dtype = tf.int32)
    lowres_height = tf.random.uniform(shape = (),
                             maxval = lowres_image_shape[0] - lowres_crop_size + 1,
                             dtype=tf.int32)

    highres_width = lowres_width * scale
    highres_height = lowres_height * scale

    lowres_cropped = lowres_image[lowres_height:lowres_height + lowres_crop_size, lowres_width:lowres_width + lowres_crop_size]
    highres_cropped = highres_image[highres_height:highres_height + highres_crop_size, highres_width:highres_width + highres_crop_size]

    return lowres_cropped, highres_cropped


def FlipImage(lowres_image, highres_image):
    random_number = tf.random.uniform(shape = (), maxval = 1)
    flipped_image = tf.cond(random_number < 0.5,
                    lambda: (lowres_image, highres_image),
                    lambda: (tf.image.flip_left_right(lowres_image),
                            tf.image.flip_left_right(highres_image)))

    return flipped_image


def RotateImage(lowres_image, highres_image):
    random_number = tf.random.uniform(shape = (), maxval = 4, dtype = tf.int32)
    rotated_image = tf.image.rot90(lowres_image, random_number), \
                    tf.image.rot90(highres_image, random_number)
    return rotated_image


####################################################################################################################################