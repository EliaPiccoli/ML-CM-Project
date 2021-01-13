import numpy as np
import utils.get_dataset as dt
from utils.model import Model
from utils.layer import Layer
from utils.plot import Plot

# ----------------------------------------- MAIN ----------------------------------------- #


train, validation, test, train_labels, validation_labels, test_labels = dt._get_split_cup()

# create model
model = Model()
model._add_layer(Layer(8, "leaky_relu", _input=(10,)))
model._add_layer(Layer(8, "leaky_relu"))
model._add_layer(Layer(8, "leaky_relu"))
model._add_layer(Layer(2, "linear"))
model._compile(eta=0.000005, loss_function="mse", alpha=0.98, _lambda=1e-1)
epoch = 50
classification=False
stats = model._train(train, train_labels, validation, validation_labels, decay=1e-6,batch_size=len(train), epoch=epoch, classification=classification, verbose=True)

# testing the model
print("Test MEE: {:.6f}".format(model._infer(test, test_labels, classification=classification)[0]))


Plot._plot_train_stats([stats], epochs=[epoch], classification = classification)