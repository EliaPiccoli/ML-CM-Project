import csv
import os
import numpy as np
import random

# -------------------------------------------------------- MLCUP -------------------------------------------------------- #
cup_path = os.path.dirname(os.path.abspath(__file__))[:-11] + '\\ml_cup\\ML-CUP20-'

def _get_cup(dataset_type='train'): # can either be 'train' or 'test'
    """ Both 'train' and 'test' set are composed of an initial index column (useless for the application),
        followed by 10 input columns.
        'train' set has 2 additional columns, for output.
    """
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
            seed = 123
            random.Random(seed).shuffle(data_set)
            random.Random(seed).shuffle(data_label)
        else: # test data
            for row in rows[7:]:
                float_row = [float(i) for i in row[1:]]
                data_set.append(float_row)
            data_label = None
    
    return np.array(data_set), np.array(data_label)