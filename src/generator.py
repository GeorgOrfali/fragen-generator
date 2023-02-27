import tensorflow as tf
from path import Path
import numpy as np
import os
import time
import json


class FalseGenerator:
    path = Path('datenset/GermanQuAD_train.json')
    train_data = []

    def train(self):
        with self.path.open(mode='rt', encoding='utf-8') as f:
            data_temp = json.load(f)
        for i in range(len(data_temp['data'])):
            paragraph = data_temp['data'][i]['paragraphs'][0]
            context = paragraph['context']
            answer = paragraph['qas'][0]['answers'][0]['text']
            c = context[(context.find("==\n") + 2):]
            self.train_data.append((answer, c))
        # after loading data, we prepare it for the training
        print(self.train_data)
        #self.prepare_training_data()

    def prepare_training_data(self,):
        vocab = sorted(set(self.train_data))
        ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
        all_ids = ids_from_chars(tf.strings.unicode_split(self.train_data, 'UTF-8'))
        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
        seq_length = 100
        sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)
