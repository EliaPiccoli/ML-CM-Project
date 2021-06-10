import numpy as np

class Mse:
    def _compute_loss(self, actual, expected):
        assert(len(actual) == len(expected))
        return (0.5 * np.sum((np.array(expected) - np.array(actual))**2)) / len(expected)

    def _compute_loss_prime(self, actual, expected): # even if multiple output this will receive one comparison at a time (look ad model.py)
        return expected - actual

def get_loss(name):
    if name == "mse":
        return Mse()
    else:
        raise Exception("Unknown loss function " + name)