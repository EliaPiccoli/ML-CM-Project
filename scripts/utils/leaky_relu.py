import numpy as np

class LeakyRelu:
    def _compute(self, input, param=0.01):
        return param*input if input < 0 else input

    def _gradient(self, input, param=0.01):
        return param if input < 0 else 1

    def __str__(self):
        return "Leaky Relu"