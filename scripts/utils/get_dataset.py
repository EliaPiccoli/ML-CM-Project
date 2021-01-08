import csv
import os
import numpy as np
import random
import copy
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------- MONK -------------------------------------------------------- #

"""remember, input is read like: [null, class, val1, val2, val3, val4, val5, val6, label]"""

dataset_path = os.path.dirname(os.path.abspath(__file__))[:-13] + 'monk_dataset/monks-'
monk_max = [3,3,2,3,4,2]

def _get_train_validation_data(dataset_num, split=0.25):
    inputs, labels = _get_dataset(dataset_num,'train')
    seed = np.random.randint(0,42069)
    random.Random(seed).shuffle(inputs)
    random.Random(seed).shuffle(labels)
    l = int(len(inputs)*split)
    # train, validation, train_labels, validation_labels
    return inputs[l:], inputs[:l], labels[l:], labels[:l]

def _get_test_data(dataset_num):
    return _get_dataset(dataset_num,'test')

def _get_dataset(dataset_num, dataset_type): #dataset_num in [1,2,3] dataset_type in ['train','test']
    # returns 2 lists, first one being dataset requested (either training or test) and second one being data label wrt dataset
    if dataset_num not in [1,2,3]:
        print('Arguments given are not acceptable!\nPossible nums: 1,2,3')
        return None, None

    data_set = list()
    data_label = list()
    path = dataset_path + str(dataset_num) + "." + dataset_type
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            int_row = [int(i) for i in row[1:8]]
            data_set.append(int_row[1:7])
            data_label.append(int_row[0])
    return data_set, data_label

def _get_one_hot_encoding(pattern):
    one_hot_pattern = []
    for i in range(len(pattern)):
        one_hot_pattern.append(np.eye(monk_max[i])[pattern[i]-1])
    return [inp for sublist in one_hot_pattern for inp in sublist]


# -------------------------------------------------------- MLCUP -------------------------------------------------------- #
cup_path = os.path.dirname(os.path.abspath(__file__))[:-13] + 'ml_cup/ML-CUP20-'

def _get_split_cup(test_split = 0.2, val_split = 0.2):
    inputs, labels = _get_cup()

    seed = np.random.randint(0,42069)
    random.Random(seed).shuffle(inputs)
    random.Random(seed).shuffle(labels)

    test_l = int(len(inputs)*test_split)
    test_inputs = copy.deepcopy(inputs[:test_l])
    test_labels = copy.deepcopy(labels[:test_l])
    train_val_inputs = inputs[test_l:]
    train_val_labels = labels[test_l:]

    scaler = MinMaxScaler()
    test_inputs = scaler.fit_transform(test_inputs)
    test_labels = scaler.fit_transform(test_labels)
    train_val_inputs = scaler.fit_transform(train_val_inputs)
    train_val_labels = scaler.fit_transform(train_val_labels)

    val_l = int(len(inputs)*val_split)

    # train, validation, test, train_labels, validation_labels, test_labels
    return train_val_inputs[val_l:], train_val_inputs[:val_l], test_inputs, train_val_labels[val_l:], train_val_labels[:val_l], test_labels


def _get_cup(dataset_type='train'): # can either be train or set
    if dataset_type not in ['train','test']:
        print('Arguments given are not acceptable!\nPossible datasets: train - test')
        return None, None

    data_set = list()
    data_label = list()
    path = cup_path + ("TR" if dataset_type == "train" else "TS") + ".csv"
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        rows = [row for row in csv_reader]
        if dataset_type == 'train':
            for row in rows[7:]:
                float_row = [float(i) for i in row[1:]]
                data_set.append(float_row[0:-2])
                data_label.append(float_row[-2:])
        else: # test data
            for row in rows[7:]:
                float_row = [float(i) for i in row[1:]]
                data_set.append(float_row)
            data_label = None
    return data_set, data_label