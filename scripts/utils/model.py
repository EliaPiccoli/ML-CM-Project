from .layer import Layer
from .loss import Loss

class Model:

    def __init__(self):
        """
        Contructor of Model class

        """
        self.layers = []

    def _add_layer(self, layer):
        self.layers.append(layer)

    def _compile(self, eta, loss_function, _lambda=0, alpha=0):
        for i in range(len(self.layers)):
            self.layers[i]._init_layer(None if i==0 else (self.layers[i-1].nodes,))
        self.eta = eta
        self._lambda = _lambda
        self.alpha = alpha
        self.loss_function_name = loss_function
        self.loss_function = Loss()._get_loss(loss_function)

    def _feed_forward(self,input):
        layer_output = input
        for i in range(len(self.layers)):
            layer_output = self.layers[i]._feed_forward(layer_output)
            print(f"Output layer{i}: {layer_output}")
        return layer_output

    def _train(self, inputs, pred):
        assert(len(inputs) == len(pred))
        outputs = []
        for i in range(len(inputs)):
            outputs.append(self._feed_forward(inputs[i]))
            loss = self.loss_function(outputs[i], pred[i]) #online case
        return loss
        #calc loss
        #update weights
        #get smart
        pass

    def __str__(self):
        result = "Model (layers: [\n\t\t"
        for layer in self.layers:
            result += str(layer) + "\n\t\t"
        result += "]\n\t"
        return result + f"eta: {self.eta}\n\tloss_function: {self.loss_function_name}\n\t_lambda: {self._lambda}\n\talpha: {self.alpha}\n\t)"


