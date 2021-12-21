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

# Importing the metric functions
from utils.metrics import *

####################################################################################################################################

LR_SIZE = 24
HR_SIZE = 96

def UpscaleBlock(x_in, num_filters):
    """
    Function Name: Upscale
    Input: Input Image Tensor x_in, Number of filters
    Description: Applies a 2D convolution of k3p1 with specified number of filters,
                 then uses the PixelShuffle function to upsample the image by a 
                 specified scale, before finally passing it through a Parametric ReLU
                 which shares the learnable parameters across space such that each filter
                 has only one set of parameters
    Output: Upsampled Image Tensor x
    """

    x = Conv2D(num_filters,
               kernel_size = 3,
               padding = 'same')(x_in)
    x = Lambda(PixelShuffle(scale = 2))(x)
    x = PReLU(shared_axes = [1,2])(x)
    
    return x

 
def ResidualBlock(x_in, num_filters, momentum = 0.8):
    """
    Function Name: ResidualBlock
    Input: Image Tensor x_in, Number of filters 
    Description: Residual Block Architecture as per Ledig et al., 2017
    """
    
    x = Conv2D(num_filters,
               kernel_size = 3,
               padding = 'same')(x_in)
    x = BatchNormalization(momentum = momentum)(x)
    x = PReLU(shared_axes = [1, 2])(x)
    x = Conv2D(num_filters, 
               kernel_size = 3,
               padding = 'same')(x)
    x = BatchNormalization(momentum = momentum)(x)
    x = Add()([x_in, x])

    return x


def SRResNet(num_filters = 64, num_res_blocks = 16):
    """
    Function Name: SRResNet
    Input: Number of filters, Number of Residual Blocks
    Description: SRResNet architecture as per Ledig et., 2017 
    """
    
    x_in = Input(shape = (None, None, 3))
    x = Lambda(NormalizeZeroToOne)(x_in)

    x = Conv2D(num_filters,
               kernel_size = 9,
               padding = 'same')(x)
    x = x_1 = PReLU(shared_axes = [1, 2])(x)
 
    # Creating the 16 Residual Blocks
    counter = 0
    while(counter < num_res_blocks):
        x = ResidualBlock(x, num_filters)
        counter += 1

    x = Conv2D(num_filters,
               kernel_size = 3,
               padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    x = UpscaleBlock(x, num_filters * 4)
    x = UpscaleBlock(x, num_filters * 4)

    x = Conv2D(3,
               kernel_size = 9,
               padding = 'same',
               activation = 'tanh')(x)
    x = Lambda(DenormalizeMinusOneToOne)(x)

    return Model(x_in, x)


generator = SRResNet

####################################################################################################################################

def DiscriminatorBlock(x_in, num_filters, strides=1, batchnorm=True, momentum=0.8):
    """
    Function Name: DiscriminatorBlock
    Input: Image Tensor x_in, Number of Filters
    Description: This function is a fundamental block which is used in the Discriminator
                 multiple times 
    Output: Image Tensor x
    """
    
    x = Conv2D(num_filters,
               kernel_size=3,
               strides = strides,
               padding = 'same')(x_in)
    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x)
    x = LeakyReLU(alpha = 0.2)(x)

    return x

def Discriminator(num_filters=64):
    """
    Function Name: Discriminator
    Input: Number of filters
    Description: 
    """
    
    x_in = Input(shape=(HR_SIZE, HR_SIZE, 3))
    x = Lambda(NormalizeMinusOneToOne)(x_in)

    x = DiscriminatorBlock(x, num_filters, batchnorm = False)
    x = DiscriminatorBlock(x, num_filters, strides = 2)
    x = DiscriminatorBlock(x, num_filters * 2)
    x = DiscriminatorBlock(x, num_filters * 2, strides = 2)
    x = DiscriminatorBlock(x, num_filters * 4)
    x = DiscriminatorBlock(x, num_filters * 4, strides = 2)
    x = DiscriminatorBlock(x, num_filters * 8)
    x = DiscriminatorBlock(x, num_filters * 8, strides = 2)

    x = Flatten()(x)

    x = Dense(1024)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Dense(1, activation = 'sigmoid')(x)

    return Model(x_in, x)


####################################################################################################################################

# Defining the VGG losses
def _vgg(output_layer):
    vgg = VGG19(input_shape = (None, None, 3), include_top = False)
    return Model(vgg.input, vgg.layers[output_layer].output) 

def vgg_22():
    return _vgg(5)


def vgg_54():
    return _vgg(20)

####################################################################################################################################