import numpy as np
import random

from .layer import Layer
from .loss import get_loss

class Model:

    def __init__(self):
        self.layers = []

    def _add_layer(self, layer):
        self.layers.append(layer)

    def _compile(self, eta=0.01, loss_function='mse', _lambda=0, alpha=0, stopping_eta=0.01, weight_range=None, weight_matrix=None, bias_matrix=None, isClassification=True, gradient_clipping=False, seed=None):
        for i in range(len(self.layers)):
            if weight_matrix is None and bias_matrix is None:
                self.layers[i]._init_layer(None if i==0 else (self.layers[i-1].nodes,), w_range=weight_range, seed=seed)
            elif weight_matrix is not None and bias_matrix is None:
                self.layers[i]._init_layer(None if i==0 else (self.layers[i-1].nodes,), weigths=weight_matrix[i], seed=seed)
            else:
                self.layers[i]._init_layer(None if i==0 else (self.layers[i-1].nodes,), weigths=weight_matrix[i], bias=bias_matrix[i], seed=seed)
        self.stopping_eta = stopping_eta*eta # useful for several lr_decay schedulers by stopping the decrease of eta at a ceratin percentage
        self.eta = eta
        self._lambda = _lambda
        self.alpha = alpha
        self.loss_function_name = loss_function
        self.loss_function = get_loss(loss_function)
        self.task = isClassification # True for Classification, False for Regression
        self.metric_function = self._compute_accuracy if isClassification else self._compute_euclidean_error
        self.grad_clip = gradient_clipping

    def _init_batch(self):
        self.batch_loss = 0
        self.batch_deltas = []
        self.batch_weights_delta = []
        for layer in self.layers:
            self.batch_deltas.append(np.zeros(layer.nodes))
            self.batch_weights_delta.append(np.zeros((layer.nodes, layer.input[0])))
    
    def _apply_decay(self, epoch_decay):
        # various other learning rate schedulers could be implemented
        self.eta = max(self.stopping_eta, self.eta/(1 + epoch_decay))

    def _init_epoch(self, epoch_decay, inp, exp):
        self.eval_metric = 0
        for layer in self.layers:
            layer.weight_delta_prev = np.zeros((layer.nodes, layer.input[0]))
            layer.bias_delta_prev = np.zeros(layer.nodes)
        self._apply_decay(epoch_decay) # update learning rate
        seed = np.random.randint(0,1000)
        random.Random(seed).shuffle(inp)
        random.Random(seed).shuffle(exp)

    def _accumulate_batch_back_prop(self, layer_bp, layer_index):
        # accumulate deltas of single pattern for batch learning
        # bias
        for i in range(len(self.batch_deltas[layer_index])):
            self.batch_deltas[layer_index][i] += layer_bp[0][i]
        # weights
        for i in range(len(self.batch_weights_delta[layer_index])):
            for j in range(len(self.batch_weights_delta[layer_index][i])):
                self.batch_weights_delta[layer_index][i][j] += layer_bp[1][i][j]

    def _feed_forward(self, _input):
        layer_output = _input
        for i in range(len(self.layers)):
            layer_output = self.layers[i]._feed_forward(layer_output)
        return layer_output

    def _back_propagation(self, expected, inp):
        # order = from output layer to input layer
        delta_last_layer = None # will be used from second iteration and on
        for i in range(len(self.layers)-1, -1, -1):
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
                multiplier = 1
                if self.grad_clip:
                    clipping_ts = 800 # to be set wrt the model behaviour (empirically obtained)
                    gradient_norm = np.linalg.norm(self.layers[i].weight_delta[j])
                    if gradient_norm > clipping_ts:
                        multiplier = (clipping_ts / gradient_norm)

                for k in range(len(self.layers[i].weights[j])):
                    # weight_n = weight_o - eta*delta(W) + alpha*delta_prev(W) - 2*lambda*weight_o
                    self.layers[i].weight_delta[j][k] *= multiplier # gradient clipping
                    self.layers[i].weight_delta_prev[j][k] = - self.eta * self.layers[i].weight_delta[j][k] + self.alpha * self.layers[i].weight_delta_prev[j][k]
                    self.layers[i].weights[j][k] = self.layers[i].weights[j][k] + self.layers[i].weight_delta_prev[j][k] - 2 * self._lambda * self.layers[i].weights[j][k]
                # bias_n = bias_o - eta*delta(W) + alpha*delta_prev(W)
                self.layers[i].bias_delta_prev[j] = - self.eta * self.layers[i].bias_delta[j] + self.alpha * self.layers[i].bias_delta_prev[j]
                self.layers[i].bias[j] = self.layers[i].bias[j] + self.layers[i].bias_delta_prev[j]

    def _ridge_regression(self):
        sum_squares = 0
        for layer in self.layers:
            for i in range(len(layer.weights)):
                for j in range(len(layer.weights[i])):
                    sum_squares += layer.weights[i][j]**2
        return self._lambda*sum_squares

    # test the model on a set of inputs, return eval_metric
    def _infer(self, inputs, expected):
        test_eval_metric = 0
        for i in range(len(inputs)):
            output = self._feed_forward(inputs[i])
            test_eval_metric = self.metric_function(output, expected[i], test_eval_metric)
        return test_eval_metric/len(inputs)

    def _validation_validation_validation(self, inputs, expected):
        self.validation_eval_metric = 0
        self.validation_loss = 0
        for i in range(len(inputs)):
            output = self._feed_forward(inputs[i])
            self.validation_eval_metric = self.metric_function(output, expected[i], self.validation_eval_metric)
            self.validation_loss += self.loss_function._compute_loss(output, expected[i])/len(inputs)
        self.validation_eval_metric /= len(inputs)

    def _train(self, train_inputs, train_expected, val_inputs, val_expected, batch_size=1, epoch=100, decay=1e-5, verbose=False):
        self.batch_size = batch_size
        self.decay_ratio = decay
        train_stats = []
        assert(len(train_inputs) == len(train_expected) and len(val_inputs) == len(val_expected))
        for e in range(epoch):
            self._init_epoch(decay*epoch, train_inputs, train_expected)
            model_epoch_loss_nr = 0.0
            for i in range(0, len(train_inputs), batch_size): # for all inputs
                j = i
                self._init_batch()
                model_batch_loss_nr = 0.0
                while j < len(train_inputs) and j-i < batch_size: # iterate over batch
                    self.model_output = self._feed_forward(train_inputs[j]) # compute prediction
                    model_batch_loss_nr += self.loss_function._compute_loss(self.model_output, train_expected[j]) # calculate loss
                    self.batch_loss += self.loss_function._compute_loss(self.model_output, train_expected[j]) + self._ridge_regression()
                    self._back_propagation(train_expected[j], train_inputs[j]) # compute back-propagation
                    self.eval_metric = self.metric_function(self.model_output, train_expected[j], self.eval_metric)
                    j += 1
                self.batch_loss = self.batch_loss / (j - i) # to avoid a bigger division on a smaller than batch size last subset of inputs
                model_epoch_loss_nr += model_batch_loss_nr / (j-i)
                self._update_layers_deltas(j - i)
                self._update_weights_bias() # update weights & bias
            self.eval_metric /= len(train_inputs)
            self._validation_validation_validation(val_inputs, val_expected)
            model_epoch_loss_nr = model_epoch_loss_nr/(len(train_inputs)/batch_size) if len(train_inputs) % batch_size == 0 else model_epoch_loss_nr/(len(train_inputs)/batch_size) + 1
            if verbose:
                print("Epoch {:4d} - LR: {:.6f} - Train_Eval_Metric: {:.6f} - Train_Loss: {:.6f} - Validation_Eval_Metric: {:.6f} - Validation_Loss: {:.6f}"
                    .format(e, self.eta, self.eval_metric, model_epoch_loss_nr, self.validation_eval_metric, self.validation_loss)) 
            train_stats.append((self.eval_metric, self.validation_eval_metric, model_epoch_loss_nr, self.validation_loss))
            
        return train_stats

    def __str__(self):
        result = "Model (layers: [\n\t\t"
        for layer in self.layers:
            result += str(layer) + "\n\t\t"
        result += "]\n\t"
        return result + f"eta: {self.eta}\n\tloss_function: {self.loss_function_name}\n\t_lambda: {self._lambda}\n\talpha: {self.alpha}\n\tbatch_size: {self.batch_size}\n\tdecay: {self.decay_ratio})"

    def _compute_accuracy(self, output, expected, current_accuracy): 
        for i in range(len(expected)):
            if abs(output[i] - expected[i]) < 0.5:
                current_accuracy += 1/len(expected)
        return current_accuracy

    def _compute_euclidean_error(self, output, expected, current_euclidean_error):
        assert(len(output) == len(expected))
        return current_euclidean_error + np.sqrt(np.sum((np.array(expected) - np.array(output))**2))