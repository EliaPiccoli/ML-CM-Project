import numpy as np
from .activation_function import AF

class Layer:

    def __init__(self, nodes, activation_function_type, bias_range=(0, 0), weights_range=(-0.69,0.69), _input=None):
        self.nodes = nodes
        self.activation_function_type = activation_function_type
        self.weights_range = weights_range
        self.bias_range = bias_range
        self.input = _input
        self.weight_delta_prev = None
        self.bias_delta_prev = None
        self.bias_delta = None
        self.weight_delta = None

    def _init_layer(self, inp=None, weigths=None, bias=None, w_range=None, seed=None):
        if inp is not None:
            self.input = inp
        if w_range is not None:
            self.weights_range = w_range
        if seed is not None:
            np.random.seed(seed)
        self.weights = np.random.uniform(self.weights_range[0], self.weights_range[1], (self.nodes, self.input[0])) if weigths is None else weigths
        self.bias = np.random.uniform(self.bias_range[0], self.bias_range[1], self.nodes) if bias is None else bias
        self.activation_function = AF[self.activation_function_type]
    
    def _feed_forward(self, inputs):
        self.output = []
        self.net = []
        for node, bias in zip(self.weights, self.bias):
            self.net.append(np.dot(node, inputs) + bias)
            self.output.append(self.activation_function._compute(self.net[-1]))
        return self.output

    def _back_propagation(self, output_prev_layer, is_output_layer=False, loss_prime_values=[], deltas_next_layer=None, weights_next_layer=None):
        weight_delta = []
        delta = []
        if is_output_layer:
            for i in range(self.nodes):
                weight_delta.append([])
                # take [-] Gradient !!
                delta.append(-loss_prime_values[i]*self.activation_function._gradient(self.net[i]))
                for j in range(len(self.weights[i])):
                    weight_delta[i].append(output_prev_layer[j] * delta[-1])
        else:
            for i in range(len(self.weights)):
                weight_delta.append([])
                weights_next_layer_j = [weight_next_layer[i] for weight_next_layer in weights_next_layer]
                delta.append(np.dot(weights_next_layer_j, deltas_next_layer)*self.activation_function._gradient(self.net[i]))
                for j in range(len(self.weights[i])):
                    weight_delta[i].append(output_prev_layer[j] * delta[-1])
        return delta, weight_delta

    def __str__(self):
        return f'Layer (nodes: {self.nodes}, activation_function: {self.activation_function_type})'