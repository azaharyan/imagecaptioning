import string
from tqdm import tqdm
import logging


logging.basicConfig(level='DEBUG')
log = logging.getLogger(__name__)

DESCRIPTION_LIMIT = 32
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
            # Skip first line which is the header
            fp.readline()

            max_length = 0
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
                        max_length = max(max_length,len(desc))
                        
                        if id not in lookup:
                            lookup[id] = list()
                        lookup[id].append(f'{START_TOKEN} {" ".join(desc)} {END_TOKEN}')
                    
        lex = set()
        for key in lookup:
            [lex.update(d.split()) for d in lookup[key]]

        self.lookup = lookup
        self.max_length = max_length + 2 # + 2 because of START and END tokens
        log.debug(f'Number of description unique words: {len(lex)}')
        log.debug(f'Number of images: {len(lookup)}')
        log.debug(f'Max length description with hard limit of {DESCRIPTION_LIMIT}: {max_length}')
        self._create_vocabulary(lex, lookup)

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

    def _create_vocabulary(self, lex, lookup):
        all_captions = []
        for image, captions in lookup.items():
            all_captions.extend(captions)
        log.debug(f'All captions: {len(all_captions)}')

        word_count = {}
        for caption in all_captions:
            for word in caption.split():
                word_count[word] = word_count.get(word, 0) + 1
        
        # Keras padding adds 0 for padding, that's why we include such word which we do not use
        # Also additional word is included for all words which count is less than the limit
        self.vocab = ['<padding>']
        self.vocab.extend([word for word in lex if word_count[word] >= WORD_OCCURANCE_LIMIT])
        self.vocab.append('<unk>')
        log.debug(f'Vocabulary size with words occurred at least {WORD_OCCURANCE_LIMIT} times: {len(self.vocab)}')
        
        self.word_to_idx = {}
        self.idx_to_word = {}
        for ind, word in enumerate(self.vocab):
            self.idx_to_word[ind] = word
            self.word_to_idx[word] = ind
