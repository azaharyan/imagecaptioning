import csv
import pickle
import os
import numpy as np
import sklearn.model_selection
import tensorflow as tf


IMAGE_SIZE = 299
NUM_CHANNELS = 3
EMBEDDING_FILE = './embeddings/glove.6B.200d.txt'
EMBEDDING_SIZE = 200
PICKLE_EMBEDDINGS = './pickles/embeddings.pickle'
PICKLE_DATASETS = './pickles/datasets.pickle'
IMAGE_PREFIX = './data/flickr30/flickr30k_images/'
RECORDS_LIMIT = 10000


class Preprocessor:

    def __init__(self):
        self.image_map = {}
        self.embedding_map = {}
        self._parse_embeddings()
        self.max_sentence = 0

    def create_image_pickle(self, image_height, image_width, num_channels):
        x_data = []
        y_data = []

        with open('./data/flickr30/results.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='|')
            errors = 0
            processed = 0
            next(csv_reader, None)
            for row in csv_reader:
                try:
                    # TODO: Currently this is not an embedding but just a tensorflow tensor in the appropriate shape
                    image_to_embedding = self._image_to_embedding(row[0])
                    sentence_to_embedding = self._sentence_to_embedding(row[2])
                    x_data.append(image_to_embedding)
                    y_data.append(sentence_to_embedding)
                except Exception as error:
                    # print(f'Invalid row skipped: {row}, error: {error}')
                    errors += 1
                finally: 
                    processed += 1
                    if (processed % 1000 == 0):
                        print(f'{processed} image-sentence pairs processed')
                    if (processed >= RECORDS_LIMIT):
                        break
        
        print(f'Valid records: {len(x_data)}')
        print(f'Invalid records: {errors}')
        self._split_and_pickle_save(x_data, y_data)

    def _split_and_pickle_save(self, x_data, y_data):
        # Train 80%, Test 10%, Validation 10%
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_data, y_data, test_size=0.2, random_state=1)
        x_test, x_valid, y_test, y_valid = sklearn.model_selection.train_test_split(x_test, y_test, test_size=0.5, random_state=1)

        print(f'Train dataset: {len(x_train)}')
        print(f'Test dataset: {len(x_test)}')
        print(f'Validation dataset: {len(x_valid)}')

        # TODO: Fix this, tensor is too large, maybe is better not to serialize the image at all but to keep it in memory when needed
        with open(PICKLE_DATASETS, 'wb') as pickle_f:
            datasets = {
                # 'x_train': x_train,
                'y_train': y_train,
                # 'x_test': x_test,
                'y_test': y_test,
                # 'x_valid': x_valid,
                'y_valid': y_valid
            }
            pickle.dump(datasets, pickle_f)
            print(f'Pickle file with all datasets saved: {PICKLE_DATASETS}')

    def _parse_embeddings(self):
        # Do NOT recreate the pickle if already exists
        if os.path.exists(PICKLE_EMBEDDINGS):
            with open(PICKLE_EMBEDDINGS, 'rb') as pickle_f:
                self.embedding_map = pickle.load(pickle_f)
        else:
            with open(EMBEDDING_FILE) as f:
                for row in f:
                    tokens = row.split(' ')
                    self.embedding_map[tokens[0]] = [float(num) for num in tokens[1:]]
        
            with open(PICKLE_EMBEDDINGS, 'wb') as pickle_f:
                pickle.dump(self.embedding_map, pickle_f)
                print(f'Pickle file with all embeddings saved: {PICKLE_EMBEDDINGS}')

        print(f'Dictionary size: {len(self.embedding_map)}')

    def _sentence_to_embedding(self, sentence):
        if sentence.count('.') == 0:
            sentence = sentence + ('.' if sentence.endswith(' ') else ' .')
        if sentence.count('.') != 1 or not sentence.endswith(' .'):
            raise Exception(f'Invalid image caption: {sentence}')
    
        embeddings = []
        for word in sentence.strip().lower().split(' '):
            embeddings.append(self._word_to_embedding(word))
        
        if len(embeddings) > self.max_sentence:
            self.max_sentence = len(embeddings)

        return embeddings
    
    def _word_to_embedding(self, word):
        if word not in self.embedding_map:
            raise Exception(f'Word not in the vocabulary: "{word}"')

        return self.embedding_map[word]

    def _image_to_embedding(self, image_file):
        img = tf.io.read_file(f'{IMAGE_PREFIX}{image_file}')
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        return img


if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor.create_image_pickle(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
