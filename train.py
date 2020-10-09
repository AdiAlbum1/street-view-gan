import tensorflow as tf
import cv2
import os
import numpy as np
import time
import random

from constants import *

from data_generator import My_Data_Generator
from models import make_generator_small_model, make_discriminator_small_model

from progressbar import ProgressBar
pbar = ProgressBar()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Load dataset to image_generator
train_image_generator = My_Data_Generator(train_path, batch_size)
val_image_generator = My_Data_Generator(val_path, batch_size)

# View dataset
if DEBUG:
    train_image_generator.visualizer()

# Define Models
generator_model = make_generator_small_model(batch_size)
discriminator_model = make_discriminator_small_model()

# Summarize Models
if DEBUG:
    print("GENERATOR:")
    print(generator_model.summary())
    print("\nDISCRIMINATOR:")
    print(discriminator_model.summary())

# # View Generator output
if DEBUG:
    noise = tf.random.normal([1, random_vector_size])
    generated_image = generator_model(noise, training=False)[0]
    generated_image = np.uint8(((generated_image + 1) * 127.5))
    generated_image = cv2.resize(generated_image, (160, 128))
    cv2.imshow("generated_image", generated_image)
    cv2.waitKey(0)


# Define loss & optimizer
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-3)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-3)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator_model,
                                 discriminator=discriminator_model)

EPOCHS = 50

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(batch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        noise, real_images = batch

        generated_images = generator_model(noise, training=True)

        real_output = discriminator_model(real_images, training=True)
        fake_output = discriminator_model(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step_generator(batch):
    with tf.GradientTape() as gen_tape:
        noise, real_images = batch

        generated_images = generator_model(noise, training=True)

        real_output = discriminator_model(real_images, training=True)
        fake_output = discriminator_model(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))

def calculate_batch_loss(batch):
    noise, real_images = batch

    generated_images = generator_model(noise, training=False)

    real_output = discriminator_model(real_images, training=False)
    fake_output = discriminator_model(generated_images, training=False)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

    return gen_loss, disc_loss

def calculate_dataset_loss(dataset):
    gen_loss = 0
    disc_loss = 0
    for batch in dataset:
        curr_gen_loss, curr_disc_loss = calculate_batch_loss(batch)
        gen_loss += curr_gen_loss
        disc_loss += curr_disc_loss

    gen_loss = gen_loss / len(dataset)
    disc_loss = disc_loss / len(dataset)

    return gen_loss, disc_loss

def train(train_dataset, val_dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        print("EPOCH #"+str(epoch+1))
        # TRAIN
        if epoch % 2 != 0:
            for batch in train_dataset:
                train_step_generator(batch)

        else:
            for batch in train_dataset:
                train_step(batch)

        # CALCULATE EPOCH LOSS
        train_gen_loss, train_disc_loss = calculate_dataset_loss(train_dataset)
        val_gen_loss, val_disc_loss = calculate_dataset_loss(val_dataset)
        print("\tTRAIN:\tgenerator_loss: ", train_gen_loss, " discriminator_loss: ", train_disc_loss)
        print("\tVAL:\tgenerator_loss: ", val_gen_loss, " discriminator_loss: ", val_disc_loss)

        # Save the model every epochs
        checkpoint.save(file_prefix = checkpoint_prefix)

        noise = tf.random.normal([1, random_vector_size])
        generated_image = generator_model(noise, training=False)[0]
        generated_image = np.uint8(((generated_image + 1) * 127.5))
        generated_image = cv2.resize(generated_image, (160, 128))
        # cv2.imshow("generated_image", generated_image)
        cv2.imwrite("gen_images//image_epoch_"+str(epoch)+".png", generated_image)
        # cv2.waitKey(0)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

print("PRE TRAINING LOSS")
train_gen_loss, train_disc_loss = calculate_dataset_loss(train_image_generator)
val_gen_loss, val_disc_loss = calculate_dataset_loss(val_image_generator)
print("\tTRAIN:\tgenerator_loss: ", train_gen_loss, " discriminator_loss: ", train_disc_loss)
print("\tVAL:\tgenerator_loss: ", val_gen_loss, " discriminator_loss: ", val_disc_loss)

print("START TRAINING")
train(train_image_generator, val_image_generator, EPOCHS)

