import numpy as np

class Softmax:
    def _compute(self, inputs, index):
        _sum = sum(np.exp(i) for i in inputs)
        return np.exp(inputs[index])/_sum

    def _gradient(self):
        # ?
        pass

    def __str__(self):
        return "SoftMax"