import tensorflow as tf
from tensorflow import keras


yolo_net = keras.models.Sequential([
    keras.layers.Input((448, 448, 3), name='inputs'),
    # ------------------------------ block01
    keras.layers.ZeroPadding2D(3, name='block01/padding'),
    keras.layers.Conv2D(64, 7, strides=2, padding='valid', name='block01/convolution'),
    keras.layers.BatchNormalization(name='block01/batch_normalization'),
    keras.layers.LeakyReLU(name='block01/leaky_relu'),
    keras.layers.MaxPooling2D(2, 2, name='block01/max_pooling'),
    # ------------------------------ block02
    keras.layers.Conv2D(192, 3, strides=1, padding='same', name='block02/convolution'),
    keras.layers.BatchNormalization(name='block02/batch_normalization'),
    keras.layers.LeakyReLU(name='block02/leaky_relu'),
    keras.layers.MaxPooling2D(2, 2, name='block02/max_pooling'),
    # ------------------------------ block03
    keras.layers.Conv2D(128, 1, strides=1, padding='same', name='block03/convolution01'),
    keras.layers.BatchNormalization(name='block03/batch_normalization01'),
    keras.layers.LeakyReLU(name='block03/leaky_relu01'),
    keras.layers.Conv2D(256, 3, strides=1, padding='same', name='block03/convolution02'),
    keras.layers.BatchNormalization(name='block03/batch_normalization02'),
    keras.layers.LeakyReLU(name='block03/leaky_relu02'),
    keras.layers.Conv2D(256, 1, strides=1, padding='same', name='block03/convolution03'),
    keras.layers.BatchNormalization(name='block03/batch_normalization03'),
    keras.layers.LeakyReLU(name='block03/leaky_relu03'),
    keras.layers.Conv2D(512, 3, strides=1, padding='same', name='block03/convolution04'),
    keras.layers.BatchNormalization(name='block03/batch_normalization04'),
    keras.layers.LeakyReLU(name='block03/leaky_relu04'),
    keras.layers.MaxPooling2D(2, 2, padding='same', name='block03/max_pooling'),
    # ------------------------------ block04
    keras.layers.Conv2D(256, 1, strides=1, padding='same'),
    keras.layers.Conv2D(512, 3, strides=1, padding='same'),
    keras.layers.Conv2D(256, 1, strides=1, padding='same'),
    keras.layers.Conv2D(512, 3, strides=1, padding='same'),
    keras.layers.Conv2D(256, 1, strides=1, padding='same'),
    keras.layers.Conv2D(512, 3, strides=1, padding='same'),
    keras.layers.Conv2D(128, 1, strides=1, padding='same'),
    keras.layers.Conv2D(512, 3, strides=1, padding='same'),
    keras.layers.Conv2D(512, 1, strides=1, padding='same'),
    keras.layers.Conv2D(1024, 3, strides=1, padding='same'),
    keras.layers.MaxPooling2D(2, 2, padding='same'),
    # # ------------------------------
    # keras.layers.Conv2D(512, 1, strides=1, padding='valid'),
    # keras.layers.Conv2D(1024, 3, strides=1, padding='valid'),
    # keras.layers.Conv2D(512, 1, strides=1, padding='valid'),
    # keras.layers.Conv2D(1024, 3, strides=1, padding='valid'),
    # keras.layers.Conv2D(1024, 3, strides=1, padding='valid'),
    # keras.layers.ZeroPadding2D(1),
    # keras.layers.Conv2D(1024, 3, strides=2, padding='valid'),
    # keras.layers.Conv2D(1024, 3, strides=1, padding='valid'),
    # keras.layers.Conv2D(1024, 3, strides=1, padding='valid'),
    # ------------------------------
], name='YoloNet')

yolo_net.summary()








