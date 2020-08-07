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
import nltk.translate.bleu_score as bleu


EMBEDDING_DIM = 200
IMAGE_OUTPUT_DIM = 2048
EPOCHS = 10
START_TOKEN = '<start>'
END_TOKEN = '<end>'
UNK_TOKEN = '<unk>'

PICKLES_FOLDER = './pickles'
MODEL_FOLDER = './model'
FLICKR_FOLDER = '../data/flickr30'
FLICKR_IMAGES_FOLDER = 'flickr30k_images'


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

    final = list(filter(lambda token: token not in {START_TOKEN, END_TOKEN, UNK_TOKEN}, in_text.split()))
    final = ' '.join(final)
    return final


def generate_test_set_captions(test_img_encoddings, word_to_idx, idx_to_word, max_length, caption_model, lookup_table):
    for image_key in list(test_img_encoddings.keys())[0:100]:
        image_path = os.path.join(FLICKR_FOLDER, FLICKR_IMAGES_FOLDER, image_key + '.jpg')
        if os.path.exists(image_path):
            image = test_img_encoddings[image_key].reshape(1, IMAGE_OUTPUT_DIM)
            x=plt.imread(image_path)
            generated_caption = generate_caption(image, word_to_idx, idx_to_word, max_length, caption_model)
            candidate = generated_caption.split()
            print("Generated caption: ", generated_caption)
            print("Original caption: ", lookup_table[image_key][0])
            references = list(map(lambda caption: caption.split(), lookup_table[image_key]))
            print(references)
            print(candidate)
            print("BLEU score: ", bleu.sentence_bleu(references, candidate))
            plt.imshow(x)
            plt.show()
            print("_____________________________________")


def generate_image_encoddings(images, images_folder):
    image_preprocessor = ImagePreprocessor()
    image_encoddings = image_preprocessor.run(images, images_folder, os.path.join(PICKLES_FOLDER, 'image_encodings_all_30k.pkl'))

    with open(os.path.join(PICKLES_FOLDER, 'datasets_30k.pkl'), 'rb') as fp:
        datasets = pickle.load(fp)
    images_test = datasets['test']
    
    np.random.shuffle(images_test)
    encodding_test = {img: image_encoddings[img] for img in tqdm(images_test) if img in image_encoddings}
    print(f'Test images: {len(encodding_test)}')

    return encodding_test


def perform_test():
    # Load vocab and images
    with open(os.path.join(PICKLES_FOLDER, 'vocab_30k.pkl'), "rb") as fp:
        vocab_info = pickle.load(fp)
    lookup = vocab_info['lookup']
    max_length = vocab_info['max_length']
    vocab_size = len(vocab_info['vocab'])

    encodding_test = generate_image_encoddings(list(lookup.keys()), os.path.join(FLICKR_FOLDER, FLICKR_IMAGES_FOLDER))

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
    model_path = os.path.join(MODEL_FOLDER, 'best_model30k.hdf5')
    caption_model.load_weights(model_path)

    generate_test_set_captions(test_img_encoddings=encodding_test,
                               word_to_idx=vocab_info['word_to_idx'],
                               idx_to_word=vocab_info['idx_to_word'],
                               max_length=max_length,
                               caption_model=caption_model,
                               lookup_table=lookup)


if __name__ == '__main__':
    perform_test()
