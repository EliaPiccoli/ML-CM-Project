import numpy as np
from numpy.core.numeric import True_
import get_cup_dataset as dt
from SVR import SVR
import time
import matplotlib.pyplot as plt
import sys
import math
import time
import sklearn.svm as ss
from sklearn.metrics import r2_score

data, data_out = dt._get_cup('train')
test_split = 0.2
test_len = int(len(data)*test_split)
test, test_out = data[:test_len, :], data_out[:test_len, :]
test_out1, test_out2 = test_out[:, 0], test_out[:, 1]
dev_set, dev_out = data[test_len:, :], data_out[test_len:, :]
dev_out1, dev_out2 = dev_out[:, 0], dev_out[:, 1]

max_iter = -1

start = time.time()
regressor = ss.SVR(kernel = 'linear', max_iter=max_iter, verbose=2)
regressor.fit(dev_set, dev_out1)
print(f"elapsed time {time.time() - start}")
y_pred = regressor.predict(test)
r2_score(test_out1, y_pred)