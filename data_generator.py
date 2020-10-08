from tensorflow.keras.utils import Sequence
import numpy as np

import cv2
# from skimage.io import imread
# from skimage.transform import resize

from constants import *

def ends_with_four(filename):
    basename = filename.split(".")[0]
    last_char = basename[-1]
    return last_char == "4"

def remove_non_forward_facing_images(filenames):
    return list(filter(ends_with_four, filenames))

class My_Data_Generator(Sequence):

    def __init__(self, image_filenames, batch_size):
        self.image_filenames = image_filenames
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_files = self.image_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_images = [cv2.imread(data_path+filename) for filename in batch_files]
        batch_images = [cv2.resize(img, image_dimensions) for img in batch_images]
        batch_images = np.array(batch_images) / 255.0

        return batch_images