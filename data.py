import numpy as np
import os

import torch
from torch.utils.data import Dataset


class Protein(Dataset):

    def __init__(self, root, seq_len):

        self.root = root
        self.prot_names = dict()
        self.labels = []
        self.indexs = []

        self.seq_len = seq_len

        self.construct_labels()

        for el in list(os.listdir(self.root)):

            key, value = el.split('_', 1)
            self.prot_names[key] = value

    def __getitem__(self, index):

        features = dict()
        index = str(index)

        name = f'{index}_{self.prot_names[index]}'
        pkl = np.load(f'{self.root}/{name}/{name}_prediction.pkl', allow_pickle=True)

        features['seq'] = self.create_sequence_matrix(pkl['seq'])
        features['ss'] = pkl['ss']
        features['matrix'] = pkl['dist']

        protein_len = features['ss'].shape[1]
        if protein_len < self.seq_len:
            features = self.padding(features)
        if protein_len > self.seq_len:
            features = self.crop(features, protein_len)
        
        features['dist_bin'] = np.zeros_like(features['matrix'][0])
        for i in range(10):
            features['dist_bin'] += pkl['dist_bin_map'][i]*features['matrix'][i]

        fa_name = f'{self.root}/{name}/{name}.fa'  # fa_name=f'{path}/{name}/{name}.fa'
        with open(fa_name) as f:
            header = str(f.readlines())
            label = int((header.split(',')[1]).split(':')[1])
            f.close()

        return (torch.tensor(features['seq'], dtype=torch.float),
                torch.tensor(features['dist_bin'], dtype=torch.float),
                torch.tensor(label, dtype=torch.long))

    def __len__(self):

        # Provide a way to get the length (number of elements) of the dataset
        return len(self.labels)

    def construct_labels(self):

        for el in list(os.listdir(self.root)):
            el_fa = el + '/' + el + '.fa'
            el = el + '/' + el + '_prediction.pkl'  # 12_asd_Asd/12_asd_Asd_prediction.pkl
            
            if os.path.isfile(self.root+'/'+el):
                self.indexs.append(el.split('_')[0])
                with open(self.root + '/' + el_fa, 'r') as f:
                    header = str(f.readlines())
                    label = int((header.split(',')[1]).split(':')[1])
                    self.labels.append(label)
                    f.close()

    def padding(self, features):
        for key, value in features.items():
            if key == 'matrix':
                features[key] = np.pad(value, ((0, 0), (0, self.seq_len - value.shape[1]), (0, self.seq_len - value.shape[2])))
            else:
                features[key] = np.pad(value, ((0, 0), (0, self.seq_len - value.shape[1])))  # 1, dict, seq_len
        return features

    def crop(self, features, protein_len):
        index_start = np.random.randint(0, protein_len - self.seq_len)
        for key, value in features.items():

            if key == 'matrix':
                features[key] = value[:, index_start:index_start+self.seq_len, index_start:index_start+self.seq_len]

            else:
                features[key] = value[:, index_start:index_start+self.seq_len]

        return features

    @staticmethod
    def create_sequence_matrix(sequence):
        encoding = {
            "A": 0, "C": 1, "D": 2,
            "E": 3, "F": 4, "G": 5,
            "H": 6, "I": 7, "K": 8,
            "L": 9, "M": 10, "N": 11,
            "P": 12, "Q": 13, "R": 14,
            "S": 15, "T": 16, "V": 17,
            "W": 18, "Y": 19,
        }
        matrix = np.zeros((20, len(sequence)))
        for index, el in enumerate(sequence):
            matrix[encoding[el]][index] = 1

        return matrix


def get_labels(root):
    labels = []
    for el in list(os.listdir(root)):
        el = el + '/' + el + '_prediction.pkl'  # 12_asd_Asd/12_asd_Asd_prediction.pkl

        if os.path.isfile(el):
            with open(root + '/' + el, 'r') as f:
                header = str(f.readlines())
                label = int((header.split(',')[1]).split(':')[1])
                labels.append(label)
                f.close()
    return labels
