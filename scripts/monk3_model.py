import numpy as np
import utils.get_dataset as dt
from utils.model import Model
from utils.layer import Layer
from utils.plot import Plot

# ----------------------------------------- MAIN ----------------------------------------- #
print("One day I will be a very smart Artificial Intelligence!")
MONK_MODEL = 3

train, validation, train_labels, validation_labels = dt._get_train_validation_data(MONK_MODEL, split=0.25)
ohe_inp = [dt._get_one_hot_encoding(i) for i in train]
ohe_val = [dt._get_one_hot_encoding(i) for i in validation]
train_exp = [[elem] for elem in train_labels]
validation_exp = [[elem] for elem in validation_labels]
test, test_labels = dt._get_test_data(MONK_MODEL)
ohe_test = [dt._get_one_hot_encoding(i) for i in test]
test_exp = [[elem] for elem in test_labels]
infos = []

# Model
model = Model()
model._add_layer(Layer(2, "tanh", _input=(17,)))
model._add_layer(Layer(2, "tanh"))
model._add_layer(Layer(1, "tanh"))
model._compile(9e-3, loss_function="mse", alpha=0.9)
epoch1 = 500
stats = model._train(ohe_inp, train_exp, ohe_val, validation_exp, batch_size=len(ohe_inp), epoch=epoch1)
infos.append(stats)
print("Test Accuracy Model 1: {:.6f}".format(model._infer(ohe_test, test_exp)[0]))

# Regularized model
model = Model()
model._add_layer(Layer(2, "tanh", _input=(17,)))
model._add_layer(Layer(2, "tanh"))
model._add_layer(Layer(1, "tanh"))
model._compile(5e-3, loss_function="mse", alpha=0.9, _lambda=1e-3)
epoch2 = 500
stats = model._train(ohe_inp, train_exp, ohe_val, validation_exp, batch_size=len(ohe_inp), epoch=epoch2)
infos.append(stats)
print("Test Accuracy Model 2: {:.6f}".format(model._infer(ohe_test, test_exp)[0]))

Plot._plot_train_stats(infos, epochs=[epoch1, epoch2])