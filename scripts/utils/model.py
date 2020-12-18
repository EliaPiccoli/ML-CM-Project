import numpy as np

from .layer import Layer
from .loss import get_loss

class Model:

    def __init__(self):
        """
        Contructor of Model class

        """
        self.layers = []

    def _add_layer(self, layer):
        self.layers.append(layer)

    def _compile(self, eta, loss_function, _lambda=0, alpha=1, weight_matrix=None, bias_matrix=None):
        for i in range(len(self.layers)):
            if weight_matrix is None and bias_matrix is None:
                self.layers[i]._init_layer(None if i==0 else (self.layers[i-1].nodes,))
            else:
                self.layers[i]._init_layer(None if i==0 else (self.layers[i-1].nodes,), weigths=weight_matrix[i], bias=bias_matrix[i])
        self.eta = eta
        self._lambda = _lambda
        self.alpha = alpha
        self.loss_function_name = loss_function
        self.loss_function = get_loss(loss_function)

    def _feed_forward(self,input):
        layer_output = input
        for i in range(len(self.layers)):
            layer_output = self.layers[i]._feed_forward(layer_output)
            #print(f"Output layer {i}: {layer_output}")
        return layer_output

    def _back_propagation(self, expected, inp):
        self.deltas = []
        for i in range(len(self.layers)-1, -1, -1):
            if i == len(self.layers)-1: # output layer
                loss_prime = []
                for j in range(len(expected)):
                    loss_prime.append(self.loss_function._compute_loss_prime(self.layer_outputs[-1][j], expected[j]))
                self.deltas.append(self.layers[i]._back_propagation(self.layers[i-1].output, is_output_layer=True, loss_prime_values=loss_prime))
            elif i == 0: # input layer
                self.deltas.append(self.layers[i]._back_propagation(inp, deltas_next_layer=self.deltas[-1], weights_next_layer=self.layers[i+1].weights))
            else: #hidden layer (where magic happens)
                self.deltas.append(self.layers[i]._back_propagation(self.layers[i-1].output, deltas_next_layer=self.deltas[-1], weights_next_layer=self.layers[i+1].weights))

    def _update_weights(self):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].weights)):
                for k in range(len(self.layers[i].weights[j])):
                    self.layers[i].weights[j][k] = self.layers[i].weights[j][k] - self.eta * self.layers[i].weight_delta[j][k] 
            #print(f"Layer {i}: {self.layers[i].weights}")

    def _train(self, inputs, expected):
        assert(len(inputs) == len(expected))
        self.layer_outputs = []
        for i in range(len(inputs)): # for all inputs
            self.layer_outputs.append(self._feed_forward(inputs[i])) # compute prediction
            self.loss = self.loss_function._compute_loss(self.layer_outputs[i], expected[i]) # calculate loss
            self._back_propagation(expected[i], inputs[i]) # compute back-propagation
            self._update_weights() # update weights
            print(f"{i} - Loss: {self.loss}")
        #get smart

    def __str__(self):
        result = "Model (layers: [\n\t\t"
        for layer in self.layers:
            result += str(layer) + "\n\t\t"
        result += "]\n\t"
        return result + f"eta: {self.eta}\n\tloss_function: {self.loss_function_name}\n\t_lambda: {self._lambda}\n\talpha: {self.alpha}\n\t)"


