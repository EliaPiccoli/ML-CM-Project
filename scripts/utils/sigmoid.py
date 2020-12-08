import numpy as np

class Sigmoid:
    def _compute(self, input):
          return 1/(1 + np.exp(-input))

    def _gradient(self):
        pass