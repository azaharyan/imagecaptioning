import logging
logging.basicConfig(level='WARNING')
log = logging.getLogger(__name__)


import numpy as np
import pickle

import os
from tqdm import tqdm
from preprocessing.image_preprocessor import ImagePreprocessor

from tensorflow.keras import Input
from tensorflow.keras.layers import (
    LSTM, Embedding, Dense, Dropout, add
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


EMBEDDING_DIM = 200
IMAGE_OUTPUT_DIM = 2048
EPOCHS = 10
START_TOKEN = '<start>'
END_TOKEN = '<end>'
UNK_TOKEN = '<unk>'


def generate_caption(image_encodding, word_to_idx, idx_to_word, max_length, caption_model):
    in_text = START_TOKEN
    for i in range(max_length):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_encodding, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word[yhat]
        in_text += ' ' + word
        if word == END_TOKEN:
            break

    # final = list(filter(lambda token: token not in {START_TOKEN, END_TOKEN, UNK_TOKEN}, in_text.split()))
    # final = ' '.join(final)
    # return final
    return in_text


def generate_test_set_captions(test_img_encoddings, word_to_idx, idx_to_word, max_length, caption_model, lookup_table):
    for image_key in list(test_img_encoddings.keys())[0:10]:
        image_path = os.path.join('../data/flickr30/flickr30k_images', image_key + '.jpg')
        if os.path.exists(image_path):
            image = test_img_encoddings[image_key].reshape(1, IMAGE_OUTPUT_DIM)
            x=plt.imread(image_path)
            print("Generated caption: ", generate_caption(image, word_to_idx, idx_to_word, max_length, caption_model))
            print("Original caption: ", lookup_table[image_key][0])
            plt.imshow(x)
            plt.show()
            print("_____________________________________")


def generate_image_encoddings(images, images_folder):
    image_preprocessor = ImagePreprocessor()
    image_encoddings = image_preprocessor.run(images, images_folder, './pickles/image_encodings_all_30k.pkl')

    # Datasets separation
    # We use the same training set all the time since the vocabulary is created based on it
    # Can be modified in the future => not to load everything when only in predict mode
    training_offset = int(len(images) * 0.8)
    images_test = images[training_offset:]
    
    np.random.shuffle(images_test)
    encodding_test = {img: image_encoddings[img] for img in tqdm(images_test) if img in image_encoddings}
    print(f'Test images: {len(encodding_test)}')

    return encodding_test


def perform_test():
    # Load vocab and images
    with open('./pickles/vocab_30k.pkl', "rb") as fp:
        vocab_info = pickle.load(fp)
    lookup = vocab_info['lookup']
    max_length = vocab_info['max_length']
    vocab_size = len(vocab_info['vocab'])

    encodding_test = generate_image_encoddings(list(lookup.keys()), '../data/flickr30/flickr30k_images')

    # Model creation
    inputs1 = Input(shape=(IMAGE_OUTPUT_DIM,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    # Mask zero = True is introduced because of the padding
    se1 = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    caption_model.compile(loss='categorical_crossentropy', optimizer='adam')
    model_path = os.path.join('./model', 'caption-model30kv2.hdf5')
    caption_model.load_weights(model_path)

    generate_test_set_captions(test_img_encoddings=encodding_test,
                               word_to_idx=vocab_info['word_to_idx'],
                               idx_to_word=vocab_info['idx_to_word'],
                               max_length=max_length,
                               caption_model=caption_model,
                               lookup_table=lookup)


if __name__ == '__main__':
    perform_test()
