import numpy as np
import logging
import os

from tqdm import tqdm
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


logging.basicConfig(level='DEBUG')
log = logging.getLogger(__name__)

EMBEDDING_DIM = 200
IMAGE_OUTPUT_DIM = 2048
EPOCHS = 10


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


def data_generator(descriptions, image_encoddings, word_to_idx, max_length, vocab_size, num_photos_per_batch):
  # x1 - Training data for photos
  # x2 - The caption that goes with each photo
  # y - The predicted rest of the caption
  x1, x2, y = [], [], []
  n=0
  while True:
    for key, desc_list in descriptions.items():
      n+=1
      photo = image_encoddings[key]
      # Each photo has several descriptions
      for desc in desc_list:
        # Convert each sentense into a list of word ids.
        seq = list(map(
            lambda word: word_to_idx[word] if word in word_to_idx else word_to_idx['<unk>'],
            desc.split())
        )
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
    
    log.debug(f'Loaded Glove embeddings, words found: {len(glove_embeddings)}')
    return glove_embeddings


def perform_training():
    # Text preprocessing
    text_prerocessor = TextPreprocessor('../data/flickr30/results.csv')
    text_prerocessor.run()
    lookup_table = text_prerocessor.get_lookup_table()
    vocab = text_prerocessor.get_vocab()
    vocab_size = len(vocab)
    word_to_idx = text_prerocessor.get_word_to_idx()
    idx_to_word = text_prerocessor.get_idx_to_word()
    max_length = text_prerocessor.get_max_length()

    # Create/load image embeddings from InceptionV3
    image_preprocessor = ImagePreprocessor()
    image_encoddings = image_preprocessor.run(lookup_table.keys(), '../data/flickr30/flickr30k_images', './pickles/image_encodings.pkl')
    
    # Datasets separation
    keys_train, keys_test = train_test_split(list(image_encoddings.keys()), test_size = 0.3)
    train_img_encoddings = {k: v for k, v in image_encoddings.items() if k in keys_train}
    test_img_encoddings = {k: v for k, v in image_encoddings.items() if k in keys_test}
    train_img_descriptions = {k: v for k, v in lookup_table.items() if k in keys_train}
    test_img_descriptions = {k: v for k, v in lookup_table.items() if k in keys_test}
    log.debug(f'Training images: {len(train_img_encoddings)}')
    log.debug(f'Test images: {len(test_img_encoddings)}')

    # Load Glove embeddings
    word_embeddings = load_glove_embeddings('../embeddings/glove.6B.200d.txt')
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in word_to_idx.items():
        embedding_vector = word_embeddings.get(word)
        if embedding_vector is not None:
            # Words not found in the glove embeddings will be all zeros, for example the <padding> word
            # Glove contains embedding for the <unk> word which is a bonus for us 
            embedding_matrix[i] = embedding_vector
    log.debug(f'Embedding matrix shape: {embedding_matrix.shape}')

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
    caption_model.summary()
    
    # Actual training
    model_path = os.path.join('./model', 'caption-model.hdf5')
    if not os.path.exists(model_path):
        number_pics_per_bath = 3
        steps = len(train_img_descriptions) // number_pics_per_bath
        for i in tqdm(range(EPOCHS * 2)):
            generator = data_generator(train_img_descriptions, train_img_encoddings, word_to_idx, max_length, vocab_size, number_pics_per_bath)
            caption_model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

        caption_model.optimizer.lr = 1e-4
        number_pics_per_bath = 6
        steps = len(train_descriptions)//number_pics_per_bath
        for i in range(EPOCHS):
            generator = data_generator(train_img_descriptions, train_img_encoddings, word_to_idx, max_length, vocab_size, number_pics_per_bath)
            caption_model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)  
        
        caption_model.save_weights(model_path)
        print(f"\Training took: {hms_string(time()-start)}")
    else:
        caption_model.load_weights(model_path)


if __name__ == '__main__':
    perform_training()
