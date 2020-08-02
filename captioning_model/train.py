import numpy as np
import logging
import os
import pickle

from tqdm import tqdm
from time import time
from preprocessing.text_preprocessor import TextPreprocessor
from preprocessing.image_preprocessor import ImagePreprocessor

from tensorflow.keras import Input
from tensorflow.keras.layers import (
    LSTM, Embedding, Dense, Dropout, add
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import glob
import nltk.translate.bleu_score as bleu


logging.basicConfig(level='WARNING')
log = logging.getLogger(__name__)


EMBEDDING_DIM = 200
IMAGE_OUTPUT_DIM = 2048
EPOCHS = 10
START_TOKEN = '<start>'
END_TOKEN = '<end>'
UNK_TOKEN = '<unk>'

PICKLES_DIR = './pickles'
FLICKR_FOLDER = '../data/flickr30'
FLICKR_IMAGES_FOLDER = 'flickr30k_images'
EMBEDDINGS_FILE = '../embeddings/glove.6B.200d.txt'
MODEL_FOLDER = './model'


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


def data_generator(descriptions, image_encoddings, word_to_idx, max_length, vocab_size, num_photos_per_batch):
  # x1 - Training data for photos
  # x2 - The caption that goes with each photo
  # y - The predicted rest of the caption
  count = 0
  x1, x2, y = [], [], []
  n=0
  while True:
    for key, desc_list in descriptions.items():
      n+=1
      photo = image_encoddings[key]
      # Each photo has several descriptions
      for desc in desc_list:
        # Convert each sentense into a list of word ids.
        # seq = list(map(
        #     lambda word: word_to_idx[word] if word in word_to_idx else word_to_idx['<unk>'],
        #     desc.split())
        # )
        seq = [word_to_idx[word] for word in desc.split(' ')]
        # Generate a training case for every possible sequence and outcome
        for i in range(1, len(seq)):
          in_seq, out_seq = seq[:i], seq[i]
          in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
          out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
          x1.append(photo)
          x2.append(in_seq)
          y.append(out_seq)
      if n==num_photos_per_batch:
        yield ([np.array(x1), np.array(x2)], np.array(y))
        x1, x2, y = [], [], []
        n=0


def load_glove_embeddings(glove_file):
    glove_embeddings = {}
    embeddings = []
    with open(glove_file, 'r') as f:
        for line in tqdm(f):
            tokens = line.split()
            word = tokens[0]
            coeffs = np.asarray(tokens[1:], dtype='float32')
            glove_embeddings[tokens[0]] = coeffs
            embeddings.append(coeffs)
    
    glove_embeddings[UNK_TOKEN] = np.mean(coeffs, axis=0)
    log.warning(f'Loaded Glove embeddings, words found: {len(glove_embeddings)}')
    return glove_embeddings


def generate_image_encoddings(images, images_folder):
    image_preprocessor = ImagePreprocessor()
    image_encoddings = image_preprocessor.run(images, images_folder, os.path.join(PICKLES_DIR, 'image_encodings_all_30k.pkl'))
    datasets_pickle_file = os.path.join(PICKLES_DIR, 'datasets_30k.pkl')

    if os.path.exists(datasets_pickle_file):
        with open(datasets_pickle_file, 'rb') as fp:
            datasets = pickle.load(fp)
            images_train, images_test, images_valid = datasets['training'], datasets['test'], datasets['validation']
    else:
        images = list(image_encoddings.keys())
        images_train, images_test_vld = train_test_split(images, train_size=0.8)
        images_test, images_valid = train_test_split(images_test_vld, train_size=0.5)

        with open(os.path.join(PICKLES_DIR, 'datasets_30k.pkl'), "wb") as fp:
            pickle.dump({
                'training': images_train,
                'test': images_test,
                'validation': images_valid
            }, fp)
    
    encodding_train = {img: enc for img, enc in tqdm(image_encoddings.items()) if img in images_train}
    encodding_test = {img: enc for img, enc in tqdm(image_encoddings.items()) if img in images_test}
    encodding_valid = {img: enc for img, enc in tqdm(image_encoddings.items()) if img in images_valid}
    log.warning(f'Training images: {len(encodding_train)}')
    log.warning(f'Test images: {len(encodding_test)}')
    log.warning(f'Validation images: {len(encodding_valid)}')

    return encodding_train, encodding_test, encodding_valid


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


def calculate_bleu_score(caption_model, dataset_encoddings, lookup_table, word_to_idx, idx_to_word, max_length):
    bleu_sum = 0
    for image_key in tqdm(list(dataset_encoddings.keys())):
        image = dataset_encoddings[image_key].reshape(1, IMAGE_OUTPUT_DIM)
        generated_caption = generate_caption(image, word_to_idx, idx_to_word, max_length, caption_model)

        candidate = generated_caption.split()
        references = list(map(lambda caption: caption.split(), lookup_table[image_key]))
        bleu_sum +=  bleu.sentence_bleu(references, candidate)
    
    return bleu_sum / len(dataset_encoddings)


def perform_training():
    # Text preprocessing
    text_preprocessor = TextPreprocessor(os.path.join(FLICKR_FOLDER, 'results.csv'))
    lookup_table = text_preprocessor.load_and_process_descriptions()

    # Create/load image embeddings from InceptionV3
    encodding_train, encodding_test, encodding_valid = generate_image_encoddings(list(lookup_table.keys()),
                                                                                 os.path.join(FLICKR_FOLDER, FLICKR_IMAGES_FOLDER))
    train_descriptions = text_preprocessor.create_training_setup(set(encodding_train.keys()))
    
    vocab = text_preprocessor.get_vocab()
    idx_to_word = text_preprocessor.get_idx_to_word()
    word_to_idx = text_preprocessor.get_word_to_idx()
    vocab_size = len(vocab)
    max_length = text_preprocessor.get_max_length()

    # Load Glove embeddings
    word_embeddings = load_glove_embeddings(EMBEDDINGS_FILE)
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in word_to_idx.items():
        embedding_vector = word_embeddings.get(word)
        if embedding_vector is not None:
            # Words not found in the glove embeddings will be all zeros, for example the <padding> word
            # Glove contains embedding for the <unk> word which is a bonus for us 
            embedding_matrix[i] = embedding_vector
        else:
            log.error(f'WORD {word} not found in Glove embeddings')
    log.warning(f'Embedding matrix shape: {embedding_matrix.shape}')

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

    caption_model.layers[2].set_weights([embedding_matrix])
    caption_model.layers[2].trainable = False
    # Default learning rate of 0.001
    caption_model.compile(loss='categorical_crossentropy', optimizer='adam')
    # caption_model.summary()
    
    # Actual training
    model_path = os.path.join(MODEL_FOLDER, 'caption-model30k.hdf5')
    if not os.path.exists(model_path):
        start = time()

        caption_model.optimizer.lr = 1e-2
        number_pics_per_bath = 3
        steps = len(train_descriptions) // number_pics_per_bath
        for i in tqdm(range(EPOCHS)):
            generator = data_generator(train_descriptions, encodding_train, word_to_idx, max_length, vocab_size, number_pics_per_bath)
            caption_model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
            caption_model.save_weights(os.path.join(MODEL_FOLDER, f'caption-model30k-snap-{i}.hdf5'))
            test_bleu_score = calculate_bleu_score(caption_model, encodding_test, lookup_table, word_to_idx, idx_to_word, max_length)
            log.warning(f'Test set BLEU score: {test_bleu_score}')

        caption_model.optimizer.lr = 1e-3
        number_pics_per_bath = 6
        steps = len(train_descriptions) // number_pics_per_bath
        for i in tqdm(range(EPOCHS)):
            generator = data_generator(train_descriptions, encodding_train, word_to_idx, max_length, vocab_size, number_pics_per_bath)
            caption_model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)  
            caption_model.save_weights(os.path.join(MODEL_FOLDER, f'caption-model30k-snap-{i}.hdf5'))
            test_bleu_score = calculate_bleu_score(caption_model, encodding_test, lookup_table, word_to_idx, idx_to_word, max_length)
            log.warning(f'Test set BLEU score: {test_bleu_score}')

        caption_model.save_weights(model_path)
        valid_bleu_score = calculate_bleu_score(caption_model, encodding_valid, lookup_table, word_to_idx, idx_to_word, max_length)
        log.warning(f'Validation set BLEU score: {valid_bleu_score}')
        log.warning(f'Training took: {hms_string(time()-start)}')
    else:
        print('Such file already exists, training aborted.')


if __name__ == '__main__':
    perform_training()
