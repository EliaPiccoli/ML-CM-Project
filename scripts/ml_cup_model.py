import numpy as np
import utils.get_dataset as dt
from utils.model import Model
from utils.layer import Layer
from utils.plot import Plot
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------- MAIN ----------------------------------------- #


train, validation, test, train_labels, validation_labels, test_labels = dt._get_split_cup()

# create model
model = Model()
model._add_layer(Layer(8, "leaky_relu", _input=(10,)))
model._add_layer(Layer(8, "leaky_relu"))
model._add_layer(Layer(8, "leaky_relu"))
model._add_layer(Layer(2, "linear"))
model._compile(eta=0.0001, loss_function="mse", alpha=0.98, _lambda=1e-2)
epoch = 50
stats = model._train(train, train_labels, validation, validation_labels, decay=1e-6,batch_size=32, epoch=epoch, verbose=True)

# testing the model
print("Test Accuracy: {:.6f}".format(model._infer(test, test_labels)[0]))


Plot._plot_train_stats([stats], epochs=[epoch], classification = False)