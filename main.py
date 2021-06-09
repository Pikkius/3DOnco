from torch.utils.data import Subset

from sklearn.model_selection import train_test_split

from data import Protein
from config import Config
from fit_eval import Train, Test


if __name__ == '__main__':
    dataset = Protein(root='Fastas')

    train_tmp_indexes, test_indexes, label_train_tmp, label_test = train_test_split(dataset.indexs, dataset.labels,
                                                                                    test_size=0.1,
                                                                                    stratify=dataset.labels)
    train_indexes, val_indexes, label_train, label_val = train_test_split(train_tmp_indexes, label_train_tmp,
                                                                          test_size=0.2, stratify=label_train_tmp)

    train_dataset = Subset(dataset, train_indexes)
    val_dataset = Subset(dataset, val_indexes)
    test_dataset = Subset(dataset, test_indexes)
    # Check dataset sizes
    print('Train Dataset: {}'.format(len(train_dataset)))
    print('Valid Dataset: {}'.format(len(val_dataset)))
    print('Test Dataset: {}'.format(len(test_dataset)))

    config=Config()

    best_net = Train(train_dataset, val_dataset, config)

    print('Accuracy test: {}'.format(Test(test_dataset, best_net, config.BATCH_SIZE)))



