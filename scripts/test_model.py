import numpy as np
import utils.get_dataset as g_dt

from utils.layer import Layer
from utils.model import Model

model = Model()
model._add_layer(Layer(3, "relu", _input=(3,)))
model._add_layer(Layer(1, "sigmoid"))
model._compile(0.25, "cross_entropy")
print(model)

for layer in model.layers:
    print("Weights\n", layer.weights)
    print("Bias\n",layer.bias)

inp = [[np.random.random() for _ in range(3)]]
exp = [np.random.randint(0,2)]
print("Input: ", inp)
print("Exp: ", exp)
print("Loss: ",model._train(inp, exp))