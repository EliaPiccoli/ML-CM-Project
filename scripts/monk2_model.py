import numpy as np
import utils.get_dataset as dt
from utils.model import Model
from utils.layer import Layer
from utils.plot import Plot

# ----------------------------------------- MAIN ----------------------------------------- #
print("One day I will be a very smart Artificial Intelligence!")
MONK_MODEL = 2

train, validation, train_labels, validation_labels = dt._get_train_validation_data(MONK_MODEL, split=0.25)
ohe_inp = [dt._get_one_hot_encoding(i) for i in train]
ohe_val = [dt._get_one_hot_encoding(i) for i in validation]
train_exp = [[elem] for elem in train_labels]
validation_exp = [[elem] for elem in validation_labels]
test, test_labels = dt._get_test_data(MONK_MODEL)
ohe_test = [dt._get_one_hot_encoding(i) for i in test]
test_exp = [[elem] for elem in test_labels]

# create model
model = Model()
model._add_layer(Layer(4, "tanh", _input=(17,)))
model._add_layer(Layer(4, "tanh"))
model._add_layer(Layer(1, "tanh"))
model._compile(eta=0.009, loss_function="mse", alpha=0.85)
epoch = 500
stats = model._train(ohe_inp, train_exp, ohe_val, validation_exp, decay=5e-6,batch_size=len(ohe_inp), epoch=epoch)

# testing the model
print("Test Accuracy: {:.6f}".format(model._infer(ohe_test, test_exp)[0]))


Plot._plot_train_stats([stats], epochs=[epoch])