import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, ReLU
from tensorflow.keras.activations import sigmoid, tanh
from constants import *

def make_generator_small_model(batch_size):
    model = tf.keras.Sequential()
    model.add(Dense(5 * 4 * 256, use_bias=False, input_shape=(random_vector_size,), batch_size=batch_size))
    model.add(BatchNormalization())

    model.add(Reshape((5, 4, 256)))
    assert model.output_shape == (batch_size, 5, 4, 256)

    model.add(Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same'))
    assert model.output_shape == (batch_size, 5, 4, 256)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'))
    assert model.output_shape == (batch_size, 10, 8, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'))
    assert model.output_shape == (batch_size, 10, 8, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
    assert model.output_shape == (batch_size, 20, 16, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same'))
    assert model.output_shape == (batch_size, 20, 16, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same'))
    assert model.output_shape == (batch_size, 40, 32, 32)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same'))
    assert model.output_shape == (batch_size, 40, 32, 32)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # model.add(Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same'))
    # assert model.output_shape == (batch_size, 80, 64, 16)
    # model.add(BatchNormalization())
    # model.add(LeakyReLU())
    #
    # model.add(Conv2DTranspose(16, (5, 5), strides=(1, 1), padding='same'))
    # assert model.output_shape == (batch_size, 80, 64, 16)
    # model.add(BatchNormalization())
    # model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', use_bias=False, activation=tanh))
    assert model.output_shape == (batch_size, small_image_dimensions[0], small_image_dimensions[1], 3)

    return model

def make_discriminator_small_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(128, (5, 5), strides=(1, 1), padding='same',
                     input_shape=[small_image_dimensions[0], small_image_dimensions[1], 3]))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    # 20, 16

    model.add(Conv2D(64, (5,5), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(32, (5,5), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    # 10, 8

    model.add(Conv2D(16, (5,5), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    # 5, 4

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation=sigmoid))

    return model