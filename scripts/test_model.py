import numpy as np
import utils.get_dataset as g_dt

from utils.layer import Layer
from utils.model import Model

model = Model()
model._add_layer(Layer(2, "sigmoid", _input=(2,)))
model._add_layer(Layer(2, "sigmoid"))
weights = [
    [
        [0.15, 0.20],
        [0.25, 0.30]
    ],
    [
        [0.40, 0.45],
        [0.50, 0.55]
    ]
]
bias = [
    [0.35, 0.35],
    [0.60, 0.60]
]
model._compile(0.5, "mse", weight_matrix=weights, bias_matrix=bias, _lambda=0.001, alpha=0.9)
# print(model)

for layer in model.layers:
    print("Weights: ", layer.weights)
    print("Bias: ",layer.bias)

inp = [[0.05, 0.10] for i in range(1000)]
exp = [[0.01, 0.99] for i in range(1000)]
# print("Input: ", inp)
# print("Exp: ", exp)
model._train(inp, exp)
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/