import numpy as np
import logging
import os

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


logging.basicConfig(level='WARNING')
log = logging.getLogger(__name__)

EMBEDDING_DIM = 200
IMAGE_OUTPUT_DIM = 2048
EPOCHS = 10
START_TOKEN = '<start>'
END_TOKEN = '<end>'
UNK_TOKEN = '<unk>'


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
        seq = [word_to_idx[word] for word in desc.split(' ') if word in word_to_idx]
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
    with open(glove_file, 'r') as f:
        for line in tqdm(f):
            tokens = line.split()
            word = tokens[0]
            coeffs = np.asarray(tokens[1:], dtype='float32')
            glove_embeddings[tokens[0]] = coeffs
    
    log.warning(f'Loaded Glove embeddings, words found: {len(glove_embeddings)}')
    return glove_embeddings

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


def generate_test_set_captions(test_img_encoddings, word_to_idx, idx_to_word, max_length, caption_model):
    for image_key in list(test_img_encoddings.keys())[0:20]:
        image_path = os.path.join('../data/flickr8/Flicker8k_Dataset/', image_key + '.jpg')
        if os.path.exists(image_path):
            image = test_img_encoddings[image_key].reshape(1, IMAGE_OUTPUT_DIM)
            x=plt.imread(image_path)
            print("Caption:", generate_caption(image, word_to_idx, idx_to_word, max_length, caption_model))
            plt.imshow(x)
            plt.show()
            print("_____________________________________")

def generate_image_encoddings():
    img = glob.glob(os.path.join('../data/flickr8/Flicker8k_Dataset/', '*.jpg'))

    train_images_path = os.path.join('../data/flickr8','Flickr8k_text','Flickr_8k.trainImages.txt') 
    train_images = set(open(train_images_path, 'r').read().strip().split('\n'))
    test_images_path = os.path.join('../data/flickr8','Flickr8k_text','Flickr_8k.testImages.txt') 
    test_images = set(open(test_images_path, 'r').read().strip().split('\n'))

    train_img = []
    test_img = []

    for i in tqdm(img):
        f = os.path.split(i)[-1]
        if f in train_images: 
            train_img.append(f.split('.')[0]) 
        elif f in test_images:
            test_img.append(f.split('.')[0])
    
    log.warning(f'TRAIN IMAGES: {len(train_img)}')
    log.warning(f'TEST IMAGES: {len(test_img)}')

    image_preprocessor = ImagePreprocessor()
    encodding_train = image_preprocessor.run(train_img, '../data/flickr8/Flicker8k_Dataset/', './pickles/image_encodings_train_8k.pkl')
    encodding_test = image_preprocessor.run(test_img, '../data/flickr8/Flicker8k_Dataset/', './pickles/image_encodings_test_8k.pkl')

    return encodding_train, encodding_test
    

def perform_training():

    # Text preprocessing
    text_preprocessor = TextPreprocessor('../data/flickr8/Flickr8k.token.txt')
    text_preprocessor.run()    

    # Create/load image embeddings from InceptionV3
    encodding_train, encodding_test = generate_image_encoddings()
    train_descriptions = text_preprocessor.create_train_descriptions(set(encodding_train.keys()))
    
    vocab = text_preprocessor.get_vocab(train_descriptions)
    idx_to_word = text_preprocessor.get_idx_to_word()
    word_to_idx = text_preprocessor.get_word_to_idx()
    vocab_size = len(vocab) + 1 # Since 0 as word id is reserved by Keras for padding
    max_length = text_preprocessor.get_max_length()

    # # Datasets separation
    # keys_train, keys_test = train_test_split(list(image_encoddings.keys()), test_size = 0.2)
    # train_img_encoddings = {k: v for k, v in tqdm(image_encoddings.items()) if k in keys_train}
    # test_img_encoddings = {k: v for k, v in tqdm(image_encoddings.items()) if k in keys_test}
    # train_img_descriptions = {k: v for k, v in tqdm(lookup_table.items()) if k in keys_train}
    # test_img_descriptions = {k: v for k, v in tqdm(lookup_table.items()) if k in keys_test}
    # log.warning(f'Training images: {len(train_img_encoddings)}')
    # log.warning(f'Training descriptions: {len(train_img_descriptions)}')
    # log.warning(f'Test images: {len(test_img_encoddings)}')
    # log.warning(f'Test descriptions: {len(test_img_descriptions)}')

    # Load Glove embeddings
    word_embeddings = load_glove_embeddings('../embeddings/glove.6B.200d.txt')
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

    # # Model creation
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
    model_path = os.path.join('./model', 'caption-model8k.hdf5')
    if not os.path.exists(model_path):
        start = time()

        number_pics_per_bath = 3
        steps = len(train_descriptions) // number_pics_per_bath
        for i in tqdm(range(EPOCHS * 2)):
            generator = data_generator(train_descriptions, encodding_train, word_to_idx, max_length, vocab_size, number_pics_per_bath)
            caption_model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

        caption_model.optimizer.lr = 1e-4
        number_pics_per_bath = 6
        steps = len(train_descriptions) // number_pics_per_bath
        for i in range(EPOCHS):
            generator = data_generator(train_descriptions, encodding_train, word_to_idx, max_length, vocab_size, number_pics_per_bath)
            caption_model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)  
        
        caption_model.save_weights(model_path)
        log.warning(f'Training took: {hms_string(time()-start)}')
    else:
        caption_model.load_weights(model_path)

    generate_test_set_captions(encodding_test, word_to_idx, idx_to_word, max_length, caption_model)


if __name__ == '__main__':
    perform_training()
