import numpy as np
import utils.get_dataset as dt
from utils.model import Model
from utils.layer import Layer

# ----------------------------------------- MAIN ----------------------------------------- #
print("One day I will be a very smart Artificial Intelligence!")

train, validation, train_labels, validation_labels = dt._get_train_validation_data(1, split=0.25)
ohe_inp = [dt._get_one_hot_encoding(i) for i in train]
ohe_val = [dt._get_one_hot_encoding(i) for i in validation]
train_exp = [[elem] for elem in train_labels]
validation_exp = [[elem] for elem in validation_labels]

# create model
model = Model()
model._add_layer(Layer(4, "tanh", _input=(17,)))
model._add_layer(Layer(1, "tanh"))
model._compile(0.05, "mse", alpha=0.75)
model._train(ohe_inp, train_exp, ohe_val, validation_exp, batch_size=4, epoch=200)