import os
import string
import pickle
import logging
import numpy as np

from time import time
from tqdm import tqdm
from PIL import Image

import tensorflow.keras.preprocessing.image

from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3


logging.basicConfig(level='DEBUG')
log = logging.getLogger(__name__)

WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


class ImagePreprocessor:

    def __init__(self):
        encode_model = InceptionV3(weights='imagenet')
        self.encode_model = Model(encode_model.input, encode_model.layers[-2].output)
        self.preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input

        encode_model.summary()

    def run(self, images, images_folder, pickle_file):
        if not os.path.exists(pickle_file):
            start = time()
            image_encoddings = {}
            for id in tqdm(images):
                image_path = os.path.join(images_folder, id + '.jpg')
                img = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(HEIGHT, WIDTH))
                image_encoddings[id] = self._encode_image(img)

            with open(pickle_file, "wb") as fp:
                pickle.dump(image_encoddings, fp)
            log.debug(f'\nGenerating image encoddings took: {hms_string(time()-start)}')
        else:
            with open(pickle_file, "rb") as fp:
                image_encoddings = pickle.load(fp)
        
        return image_encoddings

    def _encode_image(self, img):
        # Resize the image to fit into the expected input of the encoding InceptionV3
        img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
        # Convert a PIL image to a numpy array
        x = tensorflow.keras.preprocessing.image.img_to_array(img)
        # Expand to 2D array
        x = np.expand_dims(x, axis=0)
        # Perform any preprocessing needed by InceptionV3
        x = self.preprocess_input(x)
        # Call InceptionV3 to extract the smaller feature set for the image.
        x = self.encode_model.predict(x) # Get the encoding vector for the image
        # Shape to correct form to be accepted by LSTM captioning network.
        x = np.reshape(x, OUTPUT_DIM )

        return x
