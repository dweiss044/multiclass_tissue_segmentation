#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:46:58 2019

@author: dweiss
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:03:38 2019

@author: dweiss
"""

from functools import partial

from keras.layers import (Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, 
                          Conv3D, Dense, Flatten, Reshape, Lambda)
from keras.engine import Model
from keras.optimizers import Adam

import keras.backend as K

K.set_image_data_format('channels_first')

from .unet import create_convolution_block, concatenate
from ..metrics import weighted_dice_coefficient_loss

from keras_contrib.layers.normalization import instancenormalization

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def VAE_model(input_shape=(1, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=1, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function='binary_crossentropy', activation_name="relu"):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf


    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer
    
    x = instancenormalization.InstanceNormalization(axis=1)(current_layer)
    x = Activation('relu')(x)
    x = Conv3D(
        filters=16,
        kernel_size=(3, 3, 3),
        strides=2,
        padding='same',
        data_format='channels_first',
        name='Dec_VAE_VD_Conv3D')(x)
    # Regularization of latent space
    last_shape = K.int_shape(x)
    latent_dim = 64
    hidden_dim = 64
    
    #decoder_inputs = Input(shape=(128,128,64), name='encoder_op')
    flat = Flatten()(current_layer)
    hidden = Dense(hidden_dim, activation='relu')(flat)
    z_mean = Dense(latent_dim, name='z_mean')(hidden)
    z_log_var = Dense(latent_dim, name='z_log_var')(hidden)
    
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, name='z')([z_mean, z_log_var])
    
    decoder_hidden = Dense(hidden_dim, activation='relu')(z)
    decoder_flat = Dense(last_shape[1] * last_shape[2] * last_shape[3] * last_shape[4], activation='relu')(decoder_hidden)
    decoder_reshaped = Reshape((last_shape[1], last_shape[2], last_shape[3], last_shape[4]))(decoder_flat)
    
    current_layer = decoder_reshaped

    decoding_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        #concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(up_sampling, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            decoding_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))
    
    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = decoding_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    return model


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2), data_format="channels_first"):
    up_sample = UpSampling3D(size=size, data_format="channels_first")(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2



