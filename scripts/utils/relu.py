import numpy as np

class Relu:
    def _compute(self, input):
        return np.max([0, input])

    def _gradient(self, input):
        return 0 if input <= 0 else 1

    def __str__(self):
        return "Relu"