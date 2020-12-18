import numpy as np

class Sigmoid:
    def _compute(self, input):
          return 1/(1 + np.exp(-input))

    def _gradient(self, input):
        sig_inp = self._compute(input)
        return sig_inp*(1-sig_inp)

    def __str__(self):
        return "Sigmoid"