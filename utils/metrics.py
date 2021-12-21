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

# Importing visualization functions
from utils.visualize import *

####################################################################################################################################

# Defining Peak Signal to Noise Ratio
def PSNR(a, b):
    psnr_value = tf.image.psnr(a, b, max_val = 255)
    return psnr_value


# Defining the Pixel Shuffle upscaler
def PixelShuffle(scale):
    upscaled_image = lambda x : tf.nn.depth_to_space(x, scale)
    return upscaled_image


# Defining the Normalization Functions
def NormalizeZeroToOne(x):
    # Dividing each pixel value by 255 
    # to get a clipped value between [0,1]
    return x / 255.0

def NormalizeMinusOneToOne(x):
    # Dividing each pixel value by 127.5
    # and then subtracting 1 to get a 
    # clipped value between [-1,1]
    return x / 127.5 - 1

def DenormalizeMinusOneToOne(x):
    # This function is the mathematical inverse of the
    # NormalizeMinusOneToOne Function
    return (x + 1) * 127.5


# Defining the evaluating function
def Evaluate(model, dataset):
    psnr_values = []
    for lowres, highres in dataset:
        superres = Resolve(model, lowres)
        psnr_value = PSNR(highres, superres)[0]
        psnr_values.append(psnr_value)

    # Calculating the Mean PSNR 
    mean_psnr = tf.reduce_mean(psnr_values)

    return mean_psnr

####################################################################################################################################