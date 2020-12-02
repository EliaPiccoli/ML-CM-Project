class Layer:
    def __init__(self, nodes, activation_function, weights, bias, _input=None):
        self.nodes = nodes
        self.activation_function = activation_function
        self.weights = weights
        self.bias = bias
        self.input = _input

    #methods
