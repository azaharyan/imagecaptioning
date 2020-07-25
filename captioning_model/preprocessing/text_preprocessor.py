import string
from tqdm import tqdm
import logging


logging.basicConfig(level='WARNING')
log = logging.getLogger(__name__)

DESCRIPTION_LIMIT = 50
WORD_OCCURANCE_LIMIT = 10

START_TOKEN = '<start>'
END_TOKEN = '<end>'


class TextPreprocessor:
    def __init__(self, file_name):
        self.file_name = file_name
    
    def run(self):
        null_punct = str.maketrans('', '', string.punctuation)
        lookup = dict()

        with open(self.file_name, 'r') as fp:
            max_length = 0
            for line in tqdm(fp.read().splitlines()):
                sections = line.split()
                if len(sections) >= 2:
                    id = sections[0].split('.')[0]
                    desc = sections[1: ]
   
                    # Remove very short words, punctuation and clear description
                    desc = [word.lower().strip() for word in desc]
                    desc = [w.translate(null_punct) for w in desc]
                    desc = [word for word in desc if len(word) > 1]
                    desc = [word for word in desc if word.isalpha()]

                    if len(desc) <= DESCRIPTION_LIMIT:
                        max_length = max(max_length,len(desc))
                        if id not in lookup:
                            lookup[id] = list()
                        # lookup[id].append(f'{START_TOKEN} {" ".join(desc)} {END_TOKEN}')
                        lookup[id].append(" ".join(desc))
                    
        lex = set()
        for key in lookup:
            [lex.update(d.split()) for d in lookup[key]]

        self.lookup = lookup
        self.max_length = max_length
        log.warn(f'Number of description unique words: {len(lex)}')
        log.warn(f'Number of images: {len(lookup)}')
        log.warn(f'Max length description with hard limit of {DESCRIPTION_LIMIT}: {max_length}')
        # self._create_vocabulary(lex, lookup)

    def get_vocab(self, train_descriptions):
        return self._create_vocabulary(train_descriptions)

    def get_word_to_idx(self):
        return self.word_to_idx

    def get_idx_to_word(self):
        return self.idx_to_word
    
    def get_max_length(self):
        return self.max_length
    
    def get_lookup_table(self):
        return self.lookup
    
    def create_train_descriptions(self, train_images):
        train_descriptions = {k:v for k,v in self.lookup.items() if k in train_images}
        for _,v in train_descriptions.items(): 
            for d in range(len(v)):
                v[d] = f'{START_TOKEN} {v[d]} {END_TOKEN}'

        self.max_length += 2
        log.warning(f'TRAIN DESCRIPTIONS LENGTH: {len(train_descriptions)}')
        return train_descriptions

    def _create_vocabulary(self, train_descriptions):
        all_captions = []
        for image, captions in train_descriptions.items():
            all_captions.extend(captions)
        log.warning(f'All captions: {len(all_captions)}')

        word_count = {}
        for caption in all_captions:
            for word in caption.split():
                word_count[word] = word_count.get(word, 0) + 1
        
        # Keras padding adds 0 for padding, that's why we include such word which we do not use
        # Also additional word is included for all words which count is less than the limit
        # self.vocab = ['<padding>']
        vocab = [word for word in word_count if word_count[word] >= WORD_OCCURANCE_LIMIT]
        # self.vocab.append('<unk>')
        log.warning(f'Vocabulary size with words occurred at least {WORD_OCCURANCE_LIMIT} times: {len(vocab)}')
        log.warning(f'VOCAB: {vocab[0: 30]} {vocab[-30:]}')

        self.word_to_idx = {}
        self.idx_to_word = {}
        for ind, word in enumerate(vocab):
            self.idx_to_word[ind + 1] = word
            self.word_to_idx[word] = ind + 1
        
        return vocab
