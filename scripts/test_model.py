import numpy as np
import utils.get_dataset as g_dt

from utils.layer import Layer
from utils.model import Model

model = Model()
model.add_layer(Layer(3, "relu", _input=(6,)))
model.add_layer(Layer(32, "relu"))
model.add_layer(Layer(2, "softmax"))
model.compile(0.25, "hinge")
print(model)

#print("Weights\n", model[0].weights)
#print("Bias\n", model[0].bias)
#
#inp = g_dt._get_train_data(1)[0]
#out = [model[0]._feed_forward(x) for x in inp]
#
#print("Input\n", inp[0])
#print("Output\n", out[0])