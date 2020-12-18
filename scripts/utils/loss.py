import numpy as np
from math import log

class Loss:
    #-----------------------------------------------------   FOR REGRESSION   -----------------------------------------------------#

    def _mse(self, actual, expected):
        """
        Computation of Mean Squared Error loss function

        Parameters:
        actual (float list) : ground truth

        expected (float list) : output from model
        
        """
        assert(len(actual) == len(expected))
        return 0.5 * np.sum((np.array(expected) - np.array(actual))**2) / len(actual) # avg(1/2(expected - actual)**2), 1/2 for better derivation

    def _msePrime(self, actual, expected, index):
        return expected[index] - actual[index]
    
    #----------------------------------------------------- FOR CLASSIFICATION -----------------------------------------------------#

    def _cross_entropy(self, actual, expected):
        """
        Computation of Cross Entropy Loss function

        Parameters:
        expected (int) : ground truth

        actual (float) : output from model
        
        """
        actual = actual[0] # list of only one element
        if expected == 1:
            # WHAT ABOUT ADDING 1e-15 to make sure we don't get log(0)  ???
            return -log(actual) #ln
        else:
            return -log(1 - actual)

    def _cross_entropy_prime(self, actual, expected, index):
        if expected == 1:
            return -1 / actual
        else:
            return -1 / (1 - actual)


    def _hinge(self, actual, expected): # SVM <3
        """
        Computation of Hinge Loss function

        Parameters:
        actual (int) : ground truth

        expected (float) : output from model
        
        """
        return np.max(0, 1 - actual * expected)

    def _get_loss(self, name):
        if name == "mse":
            return self._mse
        elif name == "cross_entropy":
            return self._cross_entropy
        elif name == "hinge":
            return self._hinge
        else:
            raise Exception("Unknown loss function " + name)