import string
from tqdm import tqdm
import logging
import pickle
import os


logging.basicConfig(level='WARNING')
log = logging.getLogger(__name__)

DESCRIPTION_LIMIT = 50
WORD_OCCURANCE_LIMIT = 10

START_TOKEN = '<start>'
END_TOKEN = '<end>'
UNK_TOKEN = '<unk>'

PICKLES_FOLDER = './pickles'


class TextPreprocessor:
    def __init__(self, file_name):
        self.file_name = file_name
    
    def load_and_process_descriptions(self):
        null_punct = str.maketrans('', '', string.punctuation)
        lookup = dict()

        with open(self.file_name, 'r') as fp:
            # Read CSV headers
            fp.readline()
            for line in tqdm(fp.read().splitlines()):
                sections = line.split('|')
                if len(sections) >= 3:
                    id = sections[0].split('.')[0]
                    desc = sections[2].split()
   
                    # Remove very short words, punctuation and clear description
                    desc = [word.lower().strip() for word in desc]
                    desc = [w.translate(null_punct) for w in desc]
                    desc = [word for word in desc if len(word) > 1]
                    desc = [word for word in desc if word.isalpha()]

                    if len(desc) <= DESCRIPTION_LIMIT:
                        if id not in lookup:
                            lookup[id] = list()
                        lookup[id].append(" ".join(desc))

        self.lookup = lookup
        log.warning(f'Number of images: {len(lookup)}')
        return lookup

    def get_vocab(self):
        return self.vocab

    def get_word_to_idx(self):
        return self.word_to_idx

    def get_idx_to_word(self):
        return self.idx_to_word
    
    def get_max_length(self):
        return self.max_length
    
    def get_lookup_table(self):
        return self.lookup
    
    def create_training_setup(self, train_images):
        train_descriptions = {k:v for k,v in self.lookup.items() if k in train_images}
        for _,v in train_descriptions.items(): 
            for d in range(len(v)):
                v[d] = f'{START_TOKEN} {v[d]} {END_TOKEN}'
        log.warning(f'TRAIN DESCRIPTIONS LENGTH: {len(train_descriptions)}')

        self._create_vocabulary(train_descriptions)
        # Remove words not in the vocabulary and replace with UNK_TOKEN
        for _,v in train_descriptions.items(): 
            for d in range(len(v)):
                tokens = v[d].split()
                tokens = [(token if token in self.vocab else UNK_TOKEN) for token in tokens]
                v[d] = ' '.join(tokens)

        return train_descriptions

    def _create_vocabulary(self, train_descriptions):
        all_captions = []
        for image, captions in train_descriptions.items():
            all_captions.extend(captions)
        log.warning(f'All captions: {len(all_captions)}')

        word_count = {}
        max_length = 0
        for caption in all_captions:
            tokens = caption.split()
            max_length = max(max_length, len(tokens))
            for word in tokens:
                word_count[word] = word_count.get(word, 0) + 1
        
        self.max_length = max_length
        log.warning(f'Max length description with hard limit of {DESCRIPTION_LIMIT}: {max_length}')
        log.warning(f'Number of description unique words: {len(word_count)}')
        
        # Keras padding adds 0 for padding, that's why we include such word which we do not use
        # Also additional word is included for all words which count is less than the limit
        vocab = ['<padding>']
        vocab.extend([word for word in word_count if word_count[word] >= WORD_OCCURANCE_LIMIT])
        vocab.append(UNK_TOKEN)
        # self.vocab.append('<unk>')
        self.vocab = vocab
        log.warning(f'Vocabulary size with words occurred at least {WORD_OCCURANCE_LIMIT} times: {len(vocab)}')
        log.warning(f'VOCAB: {vocab[0: 30]} {vocab[-30:]}')

        self.word_to_idx = {}
        self.idx_to_word = {}
        for ind, word in enumerate(vocab):
            self.idx_to_word[ind] = word
            self.word_to_idx[word] = ind
        
        self._create_vocab_pickle()

    def _create_vocab_pickle(self):
        pickle_obj = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab': self.vocab,
            'max_length': self.max_length,
            'lookup': self.lookup
        }

        with open(os.path.join(PICKLES_FOLDER, 'vocab_30k.pkl'), "wb") as fp:
            pickle.dump(pickle_obj, fp)
        log.warning(f'VOCAB saved as pickle file!')
