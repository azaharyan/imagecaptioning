import json
import os

class DataProvider:
    def __init__(self):

        self.dataset_root = 'data/flickr30'
        self.image_root = 'data/flickr30/images'

        dataset_path = os.path.join(self.dataset_root, 'dataset.json')
        self.dataset = json.load(open(dataset_path, 'r'))


dp = DataProvider()
print(dp.dataset)