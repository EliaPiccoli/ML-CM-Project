import numpy as np
import math
import copy
from tqdm import trange

from .layer import Layer
from .loss import get_loss

import random

class Model:

    def __init__(self):
        """
        Contructor of Model class

        """
        self.layers = []
        self.best_model = None

    def _add_layer(self, layer):
        self.layers.append(layer)

    def _compile(self, eta=0.01, loss_function='mse', _lambda=0, alpha=0, weight_matrix=None, bias_matrix=None):
        for i in range(len(self.layers)):
            if weight_matrix is None and bias_matrix is None:
                self.layers[i]._init_layer(None if i==0 else (self.layers[i-1].nodes,))
            elif weight_matrix is not None:
                self.layers[i]._init_layer(None if i==0 else (self.layers[i-1].nodes,), weigths=weight_matrix[i])
            else:
                self.layers[i]._init_layer(None if i==0 else (self.layers[i-1].nodes,), weigths=weight_matrix[i], bias=bias_matrix[i])
        self.eta = eta
        self._lambda = _lambda
        self.alpha = alpha
        self.loss_function_name = loss_function
        self.loss_function = get_loss(loss_function)

    def _init_batch(self):
        self.batch_loss = 0
        self.batch_deltas = []
        self.batch_weights_delta = []
        for layer in self.layers:
            self.batch_deltas.append(np.zeros(layer.nodes))
            self.batch_weights_delta.append(np.zeros((layer.nodes, layer.input[0])))
    
    def _init_epoch(self, lr_decay, inp, exp):
        self.accuracy = 0
        for layer in self.layers:
            layer.weight_delta_prev = np.zeros((layer.nodes, layer.input[0])) # for momentum
            layer.bias_delta_prev = np.zeros(layer.nodes) # for momentum
        self.eta = self.eta / (1 + lr_decay)
        seed = np.random.randint(0,1000)
        random.Random(seed).shuffle(inp)
        random.Random(seed).shuffle(exp)

    def _accumulate_batch_back_prop(self, layer_bp, layer_index):
        # accumulate deltas of single pattern for batch learning
        for i in range(len(self.batch_deltas[layer_index])):
            self.batch_deltas[layer_index][i] += layer_bp[0][i]

        for i in range(len(self.batch_weights_delta[layer_index])):
            for j in range(len(self.batch_weights_delta[layer_index][i])):
                self.batch_weights_delta[layer_index][i][j] += layer_bp[1][i][j]

    def _feed_forward(self,input):
        layer_output = input
        for i in range(len(self.layers)):
            layer_output = self.layers[i]._feed_forward(layer_output)
            # print(f"Output layer {i}: {layer_output}")
        return layer_output

    def _back_propagation(self, expected, inp):
        # order = from output layer to input layer
        delta_last_layer = None # will be used from second iteration and on
        for i in range(len(self.layers)-1, -1, -1):
            # print("\nLayer", i)
            if i == len(self.layers)-1: # output layer
                loss_prime = []
                for j in range(len(expected)):
                    loss_prime.append(self.loss_function._compute_loss_prime(self.model_output[j], expected[j]))
                result = self.layers[i]._back_propagation(self.layers[i-1].output, is_output_layer=True, loss_prime_values=loss_prime)
            elif i == 0: # input layer
                result = self.layers[i]._back_propagation(inp, deltas_next_layer=delta_last_layer, weights_next_layer=self.layers[i+1].weights)
            else: # hidden layer (where magic happens)
                result = self.layers[i]._back_propagation(self.layers[i-1].output, deltas_next_layer=delta_last_layer, weights_next_layer=self.layers[i+1].weights)
            self._accumulate_batch_back_prop(result,i)
            delta_last_layer = result[0]

    def _update_layers_deltas(self, batch_size):
        # copy deltas value into each layer for easier later weight and bias update
        for i in range(len(self.layers)):
            self.layers[i].bias_delta = self.batch_deltas[i]
            self.layers[i].weight_delta = self.batch_weights_delta[i]

    def _update_weights_bias(self):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].weights)):
                for k in range(len(self.layers[i].weights[j])):
                    # weight_n = weight_o - eta*delta(W) + alpha*delta_prev(W) - 2*lambda*weight_o
                    self.layers[i].weight_delta_prev[j][k] = - self.eta * self.layers[i].weight_delta[j][k] + self.alpha * self.layers[i].weight_delta_prev[j][k]
                    self.layers[i].weights[j][k] = self.layers[i].weights[j][k] + self.layers[i].weight_delta_prev[j][k] - 2 * self._lambda * self.layers[i].weights[j][k]
                # bias_n = bias_o - eta*delta(W) + alpha*delta_prev(W)
                self.layers[i].bias_delta_prev[j] = - self.eta * self.layers[i].bias_delta[j] + self.alpha * self.layers[i].bias_delta_prev[j]
                self.layers[i].bias[j] = self.layers[i].bias[j] + self.layers[i].bias_delta_prev[j]
            # print(f"\nLayer {i}:\nweights = {self.layers[i].weights}\nbias = {self.layers[i].bias}")

    # TODO: why are you here? We dont need you (should delete?)
    def _ridge_regression(self):
        sum_squares = 0
        for layer in self.layers:
            for i in range(len(layer.weights)):
                for j in range(len(layer.weights[i])):
                    sum_squares += layer.weights[i][j]**2
        # print("Regularization: ",self._lambda*sum_squares)
        return self._lambda*sum_squares

    def _compute_accuracy(self, output, expected, current_accuracy): 
        for i in range(len(expected)):
            if abs(output[i] - expected[i]) < 0.5:
                current_accuracy += 1/len(expected)
        return current_accuracy

    def _infer(self, inputs, expected):
        # executed on the best version of myself
        test_accuracy = 0
        for i in range(len(inputs)):
            output = self.best_model._feed_forward(inputs[i])
            test_accuracy = self._compute_accuracy(output, expected[i], test_accuracy)
        return (test_accuracy/len(inputs), self.best_model.validation_accuracy, self.best_model.validation_loss)

    def _validation_validation_validation(self, inputs, expected):
        self.validation_accuracy = 0
        self.validation_loss = 0
        for i in range(len(inputs)):
            output = self._feed_forward(inputs[i])
            self.validation_accuracy = self._compute_accuracy(output, expected[i], self.validation_accuracy)
            self.validation_loss += self.loss_function._compute_loss(output, expected[i], regression=0)/len(inputs)
        self.validation_accuracy /= len(inputs)
        if self.best_model is None:
            self.best_model = copy.deepcopy(self)
        elif self.validation_accuracy >= self.best_model.validation_accuracy and self.validation_loss < self.best_model.validation_loss:
            del self.best_model
            self.best_model = copy.deepcopy(self)

    def _train(self, train_inputs, train_expected, val_inputs, val_expected, batch_size=1, epoch=100, decay=1e-5):
        train_stats = []
        assert(len(train_inputs) == len(train_expected) and len(val_inputs) == len(val_expected))
        for e in range(epoch):
            self._init_epoch(decay*epoch, train_inputs, train_expected)
            for i in range(0, len(train_inputs), batch_size): # for all inputs
                j = i
                self._init_batch()
                while j < len(train_inputs) and j-i < batch_size: # iterate over batch
                    self.model_output = self._feed_forward(train_inputs[j]) # compute prediction
                    self.batch_loss += self.loss_function._compute_loss(self.model_output, train_expected[j], self._ridge_regression()) # calculate loss
                    self._back_propagation(train_expected[j], train_inputs[j]) # compute back-propagation
                    self.accuracy = self._compute_accuracy(self.model_output, train_expected[j], self.accuracy)
                    j += 1
                self.batch_loss =  self.batch_loss / (j - i) # to avoid a bigger division on a smaller than batch size last subset of inputs
                self._update_layers_deltas(j - i) # 20/12/2020 19:01
                self._update_weights_bias() # update weights & bias
                # print(f"{math.ceil(i / batch_size)} / {len(inputs) // batch_size} - Loss: {self.batch_loss}")
            self.accuracy /= len(train_inputs)
            self._validation_validation_validation(val_inputs, val_expected)
            # print("Epoch {:4d} - LR: {:.6f} - Train_Accuracy: {:.6f} - Train_Loss: {:.6f} - Validation_Accuracy: {:.6f} - Validation_Loss: {:.6f}"
            #         .format(e, self.eta, self.accuracy, self.batch_loss, self.validation_accuracy, self.validation_loss)) 
            train_stats.append((self.accuracy, self.validation_accuracy, self.batch_loss, self.validation_loss))
            
            # How do i become the best version of myself ?
            # self = self.best_model
        #get smart
        return train_stats

    def __str__(self):
        result = "Model (layers: [\n\t\t"
        for layer in self.layers:
            result += str(layer) + "\n\t\t"
        result += "]\n\t"
        return result + f"eta: {self.eta}\n\tloss_function: {self.loss_function_name}\n\t_lambda: {self._lambda}\n\talpha: {self.alpha}\n\t)"