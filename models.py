import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(Dense(5 * 4 * 256, use_bias=False, input_shape=(100,)))
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

