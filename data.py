import numpy as np
import os

import torch
from torch.utils.data import Dataset


class Protein(Dataset):

    def __init__(self, root):
        self.root = root
        self.prot_names = dict()
        for el in list(os.listdir(self.root)):
            if el[-2:] == 'fa':
                key, value = el.split('_', 1)
                self.prot_names[key] = value[:-3]

        print(self.prot_names)

    def __getitem__(self, index):

        features = dict()
        index = str(index)
        name = f'{index}_{self.prot_names[index]}'
        path = self.root
        pkl = np.load(f'{path}/{name}_prediction.pkl', allow_pickle=True)
        features['seq'] = self.create_sequence_matrix(pkl['seq'])
        features['ss'] = pkl['ss']
        features['phi'] = pkl['phi']
        features['psi'] = pkl['psi']
        features['matrix'] = pkl['dist']
        # distbin = np.zeros_like(pkl['dist'][0])
        # for i in range(10):
        #    distbin += pkl['dist_bin_map'][i] * pkl['dist'][i]
        # features['matrix'] = distbin

        protein_len = features['ss'].shape[1]
        if protein_len < 3000:
            features = self.padding(features)
        if protein_len > 3000:
            features = self.crop(features, protein_len)

        fa_name = f'{path}/{name}.fa'  # fa_name=f'{path}/{name}/{name}.fa'
        with open(fa_name) as f:
            header = str(f.readlines())
            label = int((header.split(',')[1]).split(':')[1])
            f.close()

        return (torch.tensor(features['seq'], dtype=torch.float),
                torch.tensor(features['ss'], dtype=torch.float),
                torch.tensor(features['phi'], dtype=torch.float),
                torch.tensor(features['psi'], dtype=torch.float),
                torch.tensor(features['matrix'], dtype=torch.float),
                torch.tensor(label, dtype=torch.long))

    def __len__(self):

        length = int(
            len(list(os.listdir(self.root))) / 2)  # Provide a way to get the length (number of elements) of the dataset
        return length

    @staticmethod
    def padding(features):
        for key, value in features.items():
            if key == 'matrix':
                features[key] = np.pad(value, ((0, 0), (0, 3000 - value.shape[1]), (0, 3000 - value.shape[2])))
            else:
                features[key] = np.pad(value, ((0, 0), (0, 3000 - value.shape[1])))  # 1, dict, seq_len
        return features

    @staticmethod
    def crop(features, protein_len):
        index_start = np.random.randint(0, protein_len - 3000)
        for key, value in features.items():

            if key == 'matrix':
                features[key] = value[:, index_start:index_start+3000, index_start:index_start+3000]

            else:
                features[key] = value[:, index_start:index_start+3000]

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
        if el[-2:] == 'fa':
            with open(root + '/' + str(el)) as f:
                header = str(f.readlines())
                label = int((header.split(',')[1]).split(':')[1])
                labels.append(label)
                f.close()
    return labels
