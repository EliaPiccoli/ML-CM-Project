import numpy as np
from math import log

class Mse:
    #-----------------------------------------------------   FOR REGRESSION   -----------------------------------------------------#

    def _compute_loss(self, actual, expected, regression):
        """
        Computation of Mean Squared Error loss function

        Parameters:
        expected (float list) : ground truth

        actual (float list) : output from model
        
        """
        assert(len(actual) == len(expected))
        # / len(expected) for case of multiple outputs e.g. MLCup
        #print("ACTUAL:",actual,"EXPECTED:",expected)
        return (0.5 * np.sum((np.array(expected) - np.array(actual))**2)) / len(expected) + regression # avg(1/2(expected - actual)**2), 1/2 for better derivation

    def _compute_loss_prime(self, actual, expected): # even if multiple output this will receive one comparison at a time (look ad model.py)
        return expected - actual


class CrossEntropy:
    #----------------------------------------------------- FOR CLASSIFICATION -----------------------------------------------------#

    def _compute_loss(self, actual, expected, regression, threshold=1e-10):
        """
        Computation of Cross Entropy Loss function

        Parameters:
        expected (int) : ground truth

        actual (float) : output from model
        
        """
        actual = np.clip(actual[0], threshold, 1-threshold)
        if expected == 1:
            # WHAT ABOUT ADDING 1e-15 to make sure we don't get log(0)  ???
            return -log(actual) + regression #ln
        else:
            return -log(1 - actual) + regression

    def _compute_loss_prime(self, actual, expected, threshold=1e-10):
        actual = np.clip(actual, threshold, 1-threshold)
        if expected == 1:
            return -1 / actual
        else:
            return 1 / (1 - actual)


class Hinge:
    def _compute_loss(self, actual, expected): # SVM <3
        """
        Computation of Hinge Loss function

        Parameters:
        expected (int) : ground truth

        actual (float) : output from model
        
        """
        return np.max(0, 1 - actual * expected)

def get_loss(name):
    if name == "mse":
        return Mse()
    elif name == "cross_entropy":
        return CrossEntropy()
    elif name == "hinge":
        return Hinge()
    else:
        raise Exception("Unknown loss function " + name)