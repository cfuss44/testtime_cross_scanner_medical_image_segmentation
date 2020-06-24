import tensorflow as tf
import numpy as np
from tfwrapper import layers

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

## ======================================================================
## UNet for brain segmentation
## ======================================================================
def unet2D_segmentation(images, input_shape, training, nlabels): 

    with tf.compat.v1.variable_scope('segmentation'):
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # Each conv layer consists of a convovlution, followed by activation, followed by batch normalization, 
        # ====================================
        conv1_1 = tf.compat.v1.layers.conv2d(images, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1_1')
        bn1_1 = tf.compat.v1.layers.batch_normalization(conv1_1, training=training, name='bn1_1')
        conv1_2 = tf.compat.v1.layers.conv2d(bn1_1, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1_2')
        bn1_2 = tf.compat.v1.layers.batch_normalization(conv1_2, training=training, name='bn1_2')
        pool1 = tf.compat.v1.layers.max_pooling2d(bn1_2, pool_size=(2,2), strides=(2,2), padding='same', name='pool1')
    
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = tf.compat.v1.layers.conv2d(pool1, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv2_1')
        bn2_1 = tf.compat.v1.layers.batch_normalization(conv2_1, training=training, name='bn2_1')
        conv2_2 = tf.compat.v1.layers.conv2d(bn2_1, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv2_2')
        bn2_2 = tf.compat.v1.layers.batch_normalization(conv2_2, training=training, name='bn2_2')
        pool2 = tf.compat.v1.layers.max_pooling2d(bn2_2, pool_size=(2,2), strides=(2,2), padding='same', name='pool2')

        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = tf.compat.v1.layers.conv2d(pool2, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv3_1')
        bn3_1 = tf.compat.v1.layers.batch_normalization(conv3_1, training=training, name='bn3_1')
        conv3_2 = tf.compat.v1.layers.conv2d(bn3_1, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv3_2')
        bn3_2 = tf.compat.v1.layers.batch_normalization(conv3_2, training=training, name='bn3_2')
        pool3 = tf.compat.v1.layers.max_pooling2d(bn3_2, pool_size=(2,2), strides=(2,2), padding='same', name='pool3')
    
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = tf.compat.v1.layers.conv2d(pool3, filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv4_1')
        bn4_1 = tf.compat.v1.layers.batch_normalization(conv4_1, training=training, name='bn4_1')
        conv4_2 = tf.compat.v1.layers.conv2d(bn4_1, filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv4_2')
        bn4_2 = tf.compat.v1.layers.batch_normalization(conv4_2, training=training, name='bn4_2')

        # ====================================
        # Upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upconv3 = layers.reshape_like(bn4_2, size = (tf.shape(bn3_2)[1],tf.shape(bn3_2)[2]), name='upconv3')
        concat3 = tf.concat([upconv3, conv3_2], axis=3)
        conv5_1 = tf.compat.v1.layers.conv2d(concat3, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv5_1')
        bn5_1 = tf.compat.v1.layers.batch_normalization(conv5_1, training=training, name='bn5_1')
        conv5_2 = tf.compat.v1.layers.conv2d(bn5_1, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv5_2')
        bn5_2 = tf.compat.v1.layers.batch_normalization(conv5_2, training=training, name='bn5_2')

        # ====================================
        # Upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upconv2 = layers.reshape_like(bn5_2, size = (tf.shape(bn2_2)[1],tf.shape(bn2_2)[2]), name='upconv2')
        concat2 = tf.concat([upconv2, bn2_2], axis=3)
        conv6_1 = tf.compat.v1.layers.conv2d(concat2, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv6_1')
        bn6_1 = tf.compat.v1.layers.batch_normalization(conv6_1, training=training, name='bn6_1')
        conv6_2 = tf.compat.v1.layers.conv2d(bn6_1, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv6_2')
        bn6_2 = tf.compat.v1.layers.batch_normalization(conv6_2, training=training, name='bn6_2')

        # ====================================
        # Upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upconv1 = layers.reshape_like(bn6_2, size = (tf.shape(bn1_2)[1],tf.shape(bn1_2)[2]), name='upconv1')
        concat1 = tf.concat([upconv1, bn1_2], axis=3)
        conv7_1 = tf.compat.v1.layers.conv2d(concat1, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv7_1')
        bn7_1 = tf.compat.v1.layers.batch_normalization(conv7_1, training=training, name='bn7_1')
        conv7_2 = tf.compat.v1.layers.conv2d(bn7_1, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv7_2')
        bn7_2 = tf.compat.v1.layers.batch_normalization(conv7_2, training=training, name='bn7_2')

        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        pred = tf.compat.v1.layers.conv2d(bn7_2, filters=nlabels, kernel_size=1, padding='same', activation=None, name='pred')

    return pred


## ======================================================================
## Normalization network with shared parameters
## ======================================================================
def net2D_normalization(images, input_shape):

    with tf.compat.v1.variable_scope('normalization'):
        # ====================================
        # three convolutional layers
        # ====================================
        conv1 = tf.compat.v1.layers.conv2d(images, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1_shared')
        conv2 = tf.compat.v1.layers.conv2d(conv1, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv2_shared')
        conv3 = tf.compat.v1.layers.conv2d(conv2, filters=1, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv3_shared')
    
    return conv3


## ======================================================================
## Self-supervised network for rotation task
## ======================================================================
def net2D_rotation(images, input_shape, training):

    with tf.compat.v1.variable_scope('rotation'):
        # ====================================
        # 1st Conv block - a convolutional layer, followed by batch normalization and max pooling
        # ====================================
        conv1 = tf.compat.v1.layers.conv2d(images, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1')
        bn1 = tf.compat.v1.layers.batch_normalization(conv1, training=training, name='bn1')
        pool1 = tf.compat.v1.layers.max_pooling2d(bn1, pool_size=(2,2), strides=(2,2), padding='same', name='pool1')
    
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2 = tf.compat.v1.layers.conv2d(pool1, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv2')
        bn2 = tf.compat.v1.layers.batch_normalization(conv2, training=training, name='bn2')
        pool2 = tf.compat.v1.layers.max_pooling2d(bn2, pool_size=(2,2), strides=(2,2), padding='same', name='pool2')
    
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3 = tf.compat.v1.layers.conv2d(pool2, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv3')
        bn3 = tf.compat.v1.layers.batch_normalization(conv3, training=training, name='bn3')
        pool3 = tf.compat.v1.layers.max_pooling2d(bn3, pool_size=(2,2), strides=(2,2), padding='same', name='pool3')
        print(np.shape(pool3))

	    # ====================================
        # Flatten & fully connected layers
        # ====================================
        flat = tf.compat.v1.layers.flatten(pool3, name='flat')
        den1 = tf.compat.v1.layers.dense(flat,  units=64, activation=tf.nn.relu, name='den1')
        den2 = tf.compat.v1.layers.dense(den1, units=4, name='den2')

    return den2


## ======================================================================
## Self-supervised autoencoder network
## ======================================================================
def net2D_autoencoder(images, training):

    with tf.compat.v1.variable_scope('autoencoder'):
        # ====================================
        # encoding by downsampling
        # ====================================
        o = tf.compat.v1.layers.conv2d(images, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1_1')
        o = tf.compat.v1.layers.batch_normalization(o, training=training, name='bn1_1')
        o = tf.compat.v1.layers.conv2d(o, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1_2')
        o = tf.compat.v1.layers.batch_normalization(o, training=training, name='bn1_2')
        o = tf.compat.v1.layers.max_pooling2d(o, pool_size=(2,2), strides=(2,2), padding='same', name='pool1')

        o = tf.compat.v1.layers.conv2d(o, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv2_1')
        o = tf.compat.v1.layers.batch_normalization(o, training=training, name='bn2_1')
        o = tf.compat.v1.layers.conv2d(o, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv2_2')
        o = tf.compat.v1.layers.batch_normalization(o, training=training, name='bn2_2')
        o = tf.compat.v1.layers.max_pooling2d(o, pool_size=(2,2), strides=(2,2), padding='same', name='pool2')

        o = tf.compat.v1.layers.conv2d(o, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv3_1')
        o = tf.compat.v1.layers.batch_normalization(o, training=training, name='bn3_1')
        o = tf.compat.v1.layers.conv2d(o, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv3_2')
        o = tf.compat.v1.layers.batch_normalization(o, training=training, name='bn3_2')
        o = tf.compat.v1.layers.max_pooling2d(o, pool_size=(2,2), strides=(2,2), padding='same', name='pool3')

        # ====================================
        # decoding by upsampling
        # ====================================
        o = tf.compat.v1.layers.conv2d(o, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv4_1')
        o = tf.compat.v1.layers.batch_normalization(o, training=training, name='bn4_1')
        o = tf.compat.v1.layers.conv2d(o, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv4_2')
        o = tf.compat.v1.layers.batch_normalization(o, training=training, name='bn4_2')
        o = tf.keras.layers.UpSampling2D((2, 2), name='up1')(o)

        o = tf.compat.v1.layers.conv2d(o, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv5_1')
        o = tf.compat.v1.layers.batch_normalization(o, training=training, name='bn5_1')
        o = tf.compat.v1.layers.conv2d(o, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv5_2')
        o = tf.compat.v1.layers.batch_normalization(o, training=training, name='bn5_2')
        o = tf.keras.layers.UpSampling2D((2, 2), name='up2')(o)

        o = tf.compat.v1.layers.conv2d(o, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv6_1')
        o = tf.compat.v1.layers.batch_normalization(o, training=training, name='bn6_1')
        o = tf.compat.v1.layers.conv2d(o, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv6_2')
        o = tf.compat.v1.layers.batch_normalization(o, training=training, name='bn6_2')
        o = tf.keras.layers.UpSampling2D((2, 2), name='up3')(o)

        o = tf.compat.v1.layers.conv2d(o, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv7_1')
        o = tf.compat.v1.layers.batch_normalization(o, training=training, name='bn7_1')

        # sigmoid activation in output layer
        # o = tf.compat.v1.layers.conv2d(o, filters=1, kernel_size=3, padding='same', activation=tf.nn.sigmoid, name='conv7_2')

        # identity activation in output layer
        o = tf.compat.v1.layers.conv2d(o, filters=1, kernel_size=3, padding='same', name='conv7_2')
    
    return o
