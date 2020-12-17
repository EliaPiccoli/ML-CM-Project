from .layer import Layer

class Model:

    def __init__(self):
        """
        Contructor of Model class

        """
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def compile(self, eta, loss_function, _lambda=0, alpha=0):
        for i in range(len(self.layers)):
            self.layers[i]._init_layer(None if i==0 else (self.layers[i-1].input[0],))
        self.eta = eta
        self._lambda = _lambda
        self.alpha = alpha

    def __str__(self):
         
