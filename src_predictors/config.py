import os
import json
from datetime import datetime
import shutil


class Config:
    DEVICE = 'cpu'
    BATCH_SIZE = 1

    save_out = True
    save_graph = True
    LR = 0.007  # The initial Learning Rate
    MOMENTUM = 0.9  # Hyperparameter for SGD, keep this at 0.9 when using SGD
    WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default
    NUM_EPOCHS = 20  # Total number of training epochs (iterations over dataset)
    STEP_SIZE = 20
    GAMMA = 0.1
    LOG_FREQUENCY = 20

    NET = None
    netkargs= None

    out_dir = None

    def __init__(self, dictionay=None):
        if dictionay is not None:
            self.load_config(dictionay)
        if self.out_dir is None:
            setattr(self, 'out_dir', datetime.now().strftime("%d_%b_%Y_%H"))
            if os.path.exists(self.out_dir):
                shutil.rmtree(self.out_dir)
            os.makedirs(self.out_dir)

    def __getitem__(self, item):
        return getattr(self, item)

    def load_config(self, dictionary):
        for key, value in dictionary.items():
            if not callable(getattr(self, key)):
                setattr(self, key, value)
            else:
                raise AttributeError(f'Cannot overwrite an existing callable attribute,{key}')

    def save_config(self, file=None):
        dictionary = dict(
            [(a, getattr(self, a)) for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))])
        if self.out_dir is not None:
            if file is None:
                file = 'config.txt'
            with open(f'{self.out_dir}/{file}', 'w') as f:
                json.dump(dictionary, f)
        elif file is None:
            file = 'config.txt'
            with open(file, 'w') as f:
                json.dump(dictionary, f)
        else:
            with open(file, 'w') as f:
                json.dump(dictionary, f)
