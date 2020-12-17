import numpy as np
from math import log

class Loss:
    #-----------------------------------------------------   FOR REGRESSION   -----------------------------------------------------#

    def _mse(self, actual, predicted):
        """
        Computation of Mean Squared Error loss function

        Parameters:
        actual (float list) : ground truth

        predicted (float list) : output from model
        
        """
        assert(len(actual) == len(predicted))
        return np.sum((np.array(actual) - np.array(predicted))**2) / len(actual)
    
    #----------------------------------------------------- FOR CLASSIFICATION -----------------------------------------------------#

    def _cross_entropy(self, actual, expected):
        """
        Computation of Cross Entropy Loss function

        Parameters:
        expected (int) : ground truth

        actual (float) : output from model
        
        """
        actual = actual[0]
        if expected == 1:
            # WHAT ABOUT ADDING 1e-15 to make sure we don't get log(0)  ???
            return -log(actual) #ln
        else:
            return -log(1 - actual)

    def _hinge(self, actual, predicted): # SVM <3
        """
        Computation of Hinge Loss function

        Parameters:
        actual (int) : ground truth

        predicted (float) : output from model
        
        """
        return np.max(0, 1 - actual * predicted)

    def _get_loss(self, name):
        if name == "mse":
            return self._mse
        elif name == "cross_entropy":
            return self._cross_entropy
        elif name == "hinge":
            return self._hinge
        else:
            raise Exception("Unknown loss function " + name)