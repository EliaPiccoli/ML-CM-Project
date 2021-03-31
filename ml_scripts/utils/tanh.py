import numpy as np

class Tanh:
    def _compute(self, inputs):
        return (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))

    def _gradient(self, inp):
        return 1 - self._compute(inp)**2

    def __str__(self):
        return "Tanh"