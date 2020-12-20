import csv
import os
import numpy as np

"""remember, input is read like: [null, class, val1, val2, val3, val4, val5, val6, label]"""

dataset_path = os.path.dirname(os.path.abspath(__file__))[:-13] + 'monk_dataset/monks-'
monk_max = [3,3,2,3,4,2]

def _get_train_data(dataset_num):
    return _get_dataset(dataset_num,'train')

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
