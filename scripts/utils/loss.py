import numpy as np
from math import log

class Loss:
    #-----------------------------------------------------   FOR REGRESSION   -----------------------------------------------------#

    def _mse(self, actual, predicted):
        """
        Computation of Mean Squared Error loss function

        Parameters:
        actual (float list) : list of ground truths (if online list of 1 element)

        predicted (float list) : list of predicted outputs (if online list of 1 element)
        
        """
        assert(len(actual) == len(predicted))
        return np.sum((actual - predicted)**2) / actual.size
    
    #----------------------------------------------------- FOR CLASSIFICATION -----------------------------------------------------#

    def _cross_entropy(self, actual, predicted):
        """
        Computation of Cross Entropy Loss function

        Parameters:
        actual (float list) : list of ground truths (if online list of 1 element)

        predicted (float list) : list of predicted outputs (if online list of 1 element)
        
        """
        assert(len(actual) == len(predicted))
        err_sum = 0.0
        for i in range(len(actual):
            if(predicted[i] == 1):
                err_sum += -log(actual[i]) # WHAT ABOUT ADDING 1e-15 to make sure we don't get log(0)  ???
            else:
                err_sum += -log(1 - actual[i])
        return err_sum / len(actual)

    def _hinge(self, actual, predicted): # SVM <3
        """
        Computation of Hinge Loss function

        Parameters:
        actual (float list) : list of ground truths (if online list of 1 element)

        predicted (float list) : list of predicted outputs (if online list of 1 element)
        
        """
        assert(len(actual) == len(predicted))
        err_sum = 0.0
        for i in range(len(actual):
            err_sum += np.max(0, 1 - actual[i] * predicted[i])
        return err_sum / len(actual)