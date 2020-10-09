from tensorflow.keras.utils import Sequence
import numpy as np

import cv2
import os
import random

from constants import *

def ends_with_four(filename):
    basename = filename.split(".")[0]
    last_char = basename[-1]
    return last_char == "4"

def remove_non_forward_facing_images(filenames):
    return list(filter(ends_with_four, filenames))

class My_Data_Generator(Sequence):
    def __init__(self, data_path, batch_size):
        self.data_path = data_path

        # load filenames, remove non forward-facing images and shuffle
        image_filenames = os.listdir(data_path)
        image_filenames = remove_non_forward_facing_images(image_filenames)
        random.shuffle(image_filenames)
        self.image_filenames = image_filenames

        self.batch_size = batch_size

        self.n_batches = len(image_filenames) // batch_size

    # def __iter_(self):
    #     return self
    #
    # # For Python3 compatibility
    # def __next__(self):
    #     return self.next()

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        batch_files = self.image_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_images = [cv2.imread(self.data_path + filename) for filename in batch_files]
        batch_images = [cv2.resize(img, small_image_dimensions[::-1]) for img in batch_images]
        batch_images = (np.array(batch_images) - 127.5) / 127.5  # Normalize to [-1,1]

        batch_noise = np.random.normal(0, 1, (batch_size, random_vector_size))

        batch = batch_noise, batch_images

        return batch

    def visualizer(self):
        for i in range(len(self)):
            _, image_batch = self[i]
            for j in range(self.batch_size):
                curr_image = np.uint8(((image_batch[j] + 1) * 127.5))
                curr_image = cv2.resize(curr_image, (160, 128))
                cv2.imshow("images", curr_image)
                k = cv2.waitKey(0)
                if k == 27:
                    cv2.destroyAllWindows()
                    return

        cv2.destroyAllWindows()