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
            self.layers[i]._init_layer(None if i==0 else (self.layers[i-1].nodes,))
        self.eta = eta
        self._lambda = _lambda
        self.alpha = alpha
        self.loss_function = loss_function

    def __str__(self):
        result = "Model (layers: [\n\t\t"
        for layer in self.layers:
            result += str(layer) + "\n\t\t"
        result += "]\n\t"
        return result + f"eta: {self.eta}\n\tloss_function: {self.loss_function}\n\t_lambda: {self._lambda}\n\talpha: {self.alpha}\n\t)"


