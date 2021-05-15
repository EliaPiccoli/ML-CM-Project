import numpy as np
import utils.get_dataset as dt
from utils.model import Model
from utils.layer import Layer
from utils.plot import Plot

# ----------------------------------------- MAIN ----------------------------------------- #

seed=123
train, validation, test, train_labels, validation_labels, test_labels = dt._get_split_cup(seed=seed)

# create model
model = Model()
# model._add_layer(Layer(8, "relu", _input=(10,)))
# model._add_layer(Layer(8, "relu"))
# model._add_layer(Layer(8, "relu"))
# model._add_layer(Layer(8, "relu"))
# model._add_layer(Layer(8, "relu"))
# model._add_layer(Layer(2, "linear"))
# model._compile(eta=1e-5, loss_function="mse", alpha=0.9, _lambda=1e-4, isClassification = False, stopping_eta=0.05)
# epoch = 100
# stats = model._train(train, train_labels, validation, validation_labels, decay=1e-3, batch_size=len(train), epoch=epoch,verbose=True)
model._add_layer(Layer(8, "leaky_relu", _input=(10,)))
model._add_layer(Layer(8, "leaky_relu"))
model._add_layer(Layer(8, "leaky_relu"))
model._add_layer(Layer(8, "leaky_relu"))
model._add_layer(Layer(2, "linear"))
# model._add_layer(Layer(16, "tanh", _input=(10,)))
# model._add_layer(Layer(8, "tanh"))
# model._add_layer(Layer(16, "tanh"))
# model._add_layer(Layer(8, "tanh"))
# model._add_layer(Layer(16, "tanh"))
# model._add_layer(Layer(2, "linear"))
model._compile(eta=5e-6, loss_function="mse", alpha=0.8, _lambda=1e-5, isClassification=False, stopping_eta=0.1, gradient_clipping=True, seed=seed)
epoch = 400
stats = model._train(train, train_labels, validation, validation_labels, decay=1e-5, batch_size=len(train), epoch=epoch,verbose=True)

# testing the model
print("Test MEE: {:.6f}".format(model._infer(test, test_labels)))


Plot._plot_train_stats([stats], epochs=[epoch], classification = False)