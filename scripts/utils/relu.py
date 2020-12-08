import numpy as np

class Relu:
    def _compute(self, input):
        return np.max([0, input])

    def _gradient(self):
        pass