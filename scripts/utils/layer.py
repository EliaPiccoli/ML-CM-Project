import numpy as np
from .activation_function import AF

class Layer:

    def __init__(self, nodes, activation_function_type, bias_range=(-0.5, 0.5), weights_range=(-0.69,0.69), _input=None):
        """
        Contructor of Layer class

        Parameters:
        nodes (int) : number of nodes in the layer

        activation_function_type (str) : type of activation function
        
        bias_range (tuple) : range of values for node bais (min, max)
        
        weights_range (tuple) : range of values for weights (min, max)
        
        _input (tuple) : number of input nodes
        
        """
        self.nodes = nodes
        self.activation_function_type = activation_function_type
        self.weights_range = weights_range
        self.bias_range = bias_range
        self.input = _input

    def _init_layer(self, inp=None):
        if inp is not None:
            self.input = inp
        self.weights = np.random.uniform(self.weights_range[0], self.weights_range[1], (self.nodes, self.input[0]))
        self.bias = np.random.uniform(self.bias_range[0], self.bias_range[1], self.nodes)
        self.activation_function = AF[self.activation_function_type]
    
    def _feed_forward(self, inputs):
        self.output = []
        for node, bias in zip(self.weights, self.bias):
            self.output.append(self.activation_function._compute(np.dot(node, inputs) + bias))
        return self.output

    def __str__(self):
        return f'Layer (nodes: {self.nodes}, af: {self.activation_function}, input: {self.input[0]})'