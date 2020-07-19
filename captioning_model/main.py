import numpy as np
import logging

from tqdm import tqdm
from preprocessing.text_preprocessor import TextPreprocessor
from preprocessing.image_preprocessor import ImagePreprocessor


logging.basicConfig(level='DEBUG')
log = logging.getLogger(__name__)


def data_generator(descriptions, photos, wordtoidx, max_length, num_photos_per_batch):
  # x1 - Training data for photos
  # x2 - The caption that goes with each photo
  # y - The predicted rest of the caption
  x1, x2, y = [], [], []
  n=0
  while True:
    for key, desc_list in descriptions.items():
      n+=1
      photo = photos[key+'.jpg']
      # Each photo has 5 descriptions
      for desc in desc_list:
        # Convert each word into a list of sequences.
        seq = [wordtoidx[word] for word in desc.split(' ') if word in wordtoidx]
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


def perform_training():
    text_prerocessor = TextPreprocessor('../data/flickr30/results.csv')
    text_prerocessor.run()
    lookup_table = text_prerocessor.get_lookup_table()

    image_preprocessor = ImagePreprocessor()
    images = image_preprocessor.run(lookup_table.keys(), '../data/flickr30/flickr30k_images', './pickles/image_encodings.pkl')
    
    word_embeddings = load_glove_embeddings('../embeddings/glove.6B.200d.txt')


if __name__ == '__main__':
    perform_training()
