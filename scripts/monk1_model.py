import numpy as np
import utils.get_dataset as dt

# ----------------------------------------- MAIN ----------------------------------------- #
print("One day I will be a very smart Artificial Intelligence!")

inp, exp = dt._get_train_data(1)
ohe_inp = [dt._get_one_hot_encoding(i) for i in inp]

# create model