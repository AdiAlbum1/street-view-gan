import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, ReLU
from tensorflow.keras.activations import sigmoid, tanh
from constants import *

def make_generator_small_model(batch_size):
    model = tf.keras.Sequential()
    model.add(Dense(5 * 4 * 64, use_bias=False, input_shape=(random_vector_size,), batch_size=batch_size))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Reshape((5, 4, 64)))
    assert model.output_shape == (batch_size, 5, 4, 64)

    # model.add(Conv2DTranspose(64, (3,3), strides=(1,1), padding='same', use_bias=False))
    # assert model.output_shape == (batch_size, 5, 4, 64)
    # model.add(BatchNormalization())
    # model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (batch_size, 10, 8, 64)
    model.add(BatchNormalization())
    model.add(ReLU())

    # model.add(Conv2DTranspose(32, (3,3), strides=(1,1), padding='same', use_bias=False))
    # assert model.output_shape == (batch_size, 10, 8, 32)
    # model.add(BatchNormalization())
    # model.add(LeakyReLU())

    model.add(Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (batch_size, 20, 16, 32)
    model.add(BatchNormalization())
    model.add(ReLU())

    # model.add(Conv2DTranspose(16, (3,3), strides=(1,1), padding='same', use_bias=False))
    # assert model.output_shape == (batch_size, 20, 16, 16)
    # model.add(BatchNormalization())
    # model.add(LeakyReLU())

    model.add(Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (batch_size, 40, 32, 16)
    model.add(BatchNormalization())
    model.add(ReLU())

    # model.add(Conv2DTranspose(8, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    # assert model.output_shape == (batch_size, 40, 32, 8)
    # model.add(BatchNormalization())
    # model.add(LeakyReLU())

    model.add(Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (batch_size, 80, 64, 8)
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', use_bias=False, activation=tanh))
    assert model.output_shape == (batch_size, small_image_dimensions[0], small_image_dimensions[1], 3)

    return model

def make_discriminator_small_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(8, (3, 3), strides=(2, 2), padding='same',
                     input_shape=[small_image_dimensions[0], small_image_dimensions[1], 3]))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(16, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(32, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1, activation=sigmoid))

    return model


def make_generator_large_model():
    model = tf.keras.Sequential()
    model.add(Dense(5 * 4 * 256, use_bias=False, input_shape=(random_vector_size,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((5, 4, 256)))
    assert model.output_shape == (None, 5, 4, 256)  # Note: None is the batch size

    model.add(Conv2DTranspose(128, (3,3), strides=(1,1), padding='same', use_bias=False))
    assert model.output_shape == (None, 5, 4, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 10, 8, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (3,3), strides=(1,1), padding='same', use_bias=False))
    assert model.output_shape == (None, 10, 8, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 20, 16, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(32, (3,3), strides=(1,1), padding='same', use_bias=False))
    assert model.output_shape == (None, 20, 16, 32)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 40, 32, 32)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 40, 32, 16)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 80, 64, 16)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(8, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 80, 64, 8)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 160, 128, 8)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(4, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 160, 128, 4)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(4, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 320, 256, 4)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 320, 256, 3)

    return model

