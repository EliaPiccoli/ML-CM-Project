import numpy as np
from utils.layer import Layer
import utils.get_dataset as g_dt

model = []
model.append(Layer(3, "relu", _input=(6,)))
model[0]._init_layer()

print("Weights\n", model[0].weights)
print("Bias\n", model[0].bias)

inp = g_dt._get_train_data(1)[0]
out = [model[0]._feed_forward(x) for x in inp]

print("Input\n", inp[0])
print("Output\n", out[0])