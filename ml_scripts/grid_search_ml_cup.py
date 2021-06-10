import numpy as np # more important than "#include <stdio.h>"
import utils.get_dataset as dt
import os
import pickle
import math

from utils.layer import Layer
from utils.model import Model
from utils.plot import Plot
from joblib import Parallel, delayed

class GridSearch:
    def __init__(self):
        self.eta = [0.01]
        self.alpha = [0]
        self._lambda = [0]
        self.batch_size = [1]
        self.models_layers = [] # [[Layers#1], [Layers#2], ...]
        self.lr_decay = [1e-5]
        self.epoch = [5]
        self.weight_range = [(-0.69, 0.69)]

    def _set_parameters(self, **parameters):
        if "eta" in parameters:
            self.eta = parameters["eta"]                  
        if "alpha" in parameters:
            self.alpha = parameters["alpha"]         
        if "_lambda" in parameters:
            self._lambda = parameters["_lambda"]                   
        if "batch_size" in parameters:
            self.batch_size = parameters["batch_size"]                 
        if "layers" in parameters:
            self.models_layers = parameters["layers"]               
        if "lr_decay" in parameters:
            self.lr_decay = parameters["lr_decay"]                     
        if "epoch" in parameters:
            self.epoch = parameters["epoch"]                                                                                                                         
        if "weight_range" in parameters:
            self.weight_range = parameters["weight_range"]
    
    def _compute_model_score(self, model_infos):
        # model_infos : (vacc_bm, vlossbm, training_bm[(a, va, l, vl)])
        if np.isnan(model_infos[0]):
            return 1e4
        score = 100*model_infos[0] # FOR ML CUP: try to start low and each oscillation will higher the score, then sort increasingly
        # validation loss smooth and training loss smooth (val has more weight)
        val_loss = []
        train_loss = []
        threshold = 0.75 # empirical data
        for epoch in model_infos[-2]:
            val_loss.append(epoch[3])
            train_loss.append(epoch[2])
        not_decrease_times = 0
        for i in range(len(val_loss)-1):
            if val_loss[i+1] - val_loss[i] > threshold:
                not_decrease_times += 1
        score += not_decrease_times*8

        not_decrease_times = 0
        for i in range(len(train_loss)-1):
            if train_loss[i+1] - train_loss[i] > threshold:
                not_decrease_times += 1
        score += not_decrease_times

        return score

    def _train_model(self, index, model, train, train_label, validation, validation_label, batch_size, epoch, decay):
        train_result = model._train(train, train_label, validation, validation_label, batch_size=batch_size, epoch=epoch, decay=decay)
        save_file = f"models/cup_conf{index}"
        with open(save_file, 'wb') as f:
            pickle.dump({"model": model, "layers": model.layers}, f, protocol=pickle.HIGHEST_PROTOCOL)
        return train_result

    def _run(self, train, train_label, validation, validation_label, familyofmodelsperconfiguration=1):
        print("(GS - MLCUP) - Generating weights")
        weights_per_configuration = [] # confs: [ weight_range_inits:[ weight_inits: [particular weight matrix]]]
        for configuration in self.models_layers:
            dimensions = [] # [(in, out), (in, out), ...] for each layer
            for layer in configuration:
                if len(dimensions) == 0:
                    dimensions.append((layer.input[0], layer.nodes))
                else:
                    dimensions.append((dimensions[-1][1], layer.nodes))
            for w_range in self.weight_range:            
                weight_inits = []
                for _ in range(familyofmodelsperconfiguration):
                    weight_init = []
                    for inp,out in dimensions: # for each layer create matrix weight
                        weight_init.append(np.random.uniform(w_range[0], w_range[1], (out, inp)))
                    weight_inits.append(weight_init)
                weights_per_configuration.append(weight_inits)

        # will use range(max(len, 1)) so if any value for whatever list was not provided it will iterate just one time using the default value
        # max is useless but is more clear what happens if the hyperparameter was not considered
        # if missing value the class initialize all the lists to the default value
        # list of tuple (epoch, batch, decay, compiled_model)
        print("(GS - MLCUP) - Generating models")
        models_configurations = []
        counter = 0
        for i in range(len(weights_per_configuration)):
            for j in range(len(weights_per_configuration[i])):
                for epoch_index in range(max(len(self.epoch), 1)):
                    for batch_size_index in range(max(len(self.batch_size), 1)):
                        for decay_index in range(max(len(self.lr_decay), 1)):
                            for eta_index in range(max(len(self.eta), 1)):
                                for alpha_index in range(max(len(self.alpha), 1)):
                                    for lambda_index in range(max(len(self._lambda), 1)):
                                        # initialize model
                                        model = Model()
                                        weights_matrix = []
                                        for k in range(len(weights_per_configuration[i][j])):
                                            model_layer = self.models_layers[counter//(len(self.weight_range)*familyofmodelsperconfiguration)][k]
                                            layer = Layer(model_layer.nodes, model_layer.activation_function_type, _input=model_layer.input)
                                            model._add_layer(layer)
                                            weights_copy = []
                                            for node_weights in weights_per_configuration[i][j][k]:
                                                weights_copy.append([])
                                                for weight in node_weights:
                                                    weights_copy[-1].append(weight)
                                            weights_matrix.append(weights_copy)
                                        model._compile(eta=self.eta[eta_index], alpha=self.alpha[alpha_index], _lambda=self._lambda[lambda_index], weight_matrix=weights_matrix, isClassification = False, gradient_clipping=True)
                                        models_configurations.append((self.epoch[epoch_index], self.batch_size[batch_size_index], self.lr_decay[decay_index], model))
                counter += 1
        print(f"(GS - MLCUP) - Generated {len(models_configurations)} different models.")
        print("(GS - MLCUP) - Starting Models Analysis")
        models_per_structure = len(models_configurations) // len(self.models_layers)
        configurations_per_model = len(self.epoch)*len(self.batch_size)*len(self.lr_decay)*len(self.eta)*len(self.alpha)*len(self._lambda)

        subprocess_pool_size = min(os.cpu_count(), models_per_structure)
        structures_best_configurations = []
        for i in range(len(self.models_layers)):
            print("(GS - MLCUP) - Model ", i)
            configuration_best_model = [None]*configurations_per_model

            models_training_stats = []      # [[(acc, vacc, loss, vloss), (acc, vacc, loss, vloss)], ...]
            with Parallel(n_jobs=subprocess_pool_size, verbose=10) as processes:
                result = processes(delayed(self._train_model)(i*models_per_structure+j, models_configurations[i*models_per_structure + j][3], train, train_label, validation, validation_label, models_configurations[i*models_per_structure + j][1], models_configurations[i*models_per_structure + j][0], models_configurations[i*models_per_structure + j][2]) for j in range(models_per_structure))
            
            for res in result:
                models_training_stats.append(res)

            for j in range(models_per_structure):
                training_stats = models_training_stats[j]
                best_model_vmee = training_stats[-1][1]
                best_model_vloss = training_stats[-1][3]
                index = i*models_per_structure + j
                data = {}
                with open(f"models/cup_conf{index}", 'rb') as f:
                    data = pickle.load(f)
                model = data['model']
                
                if configuration_best_model[j%configurations_per_model] is None:
                    configuration_best_model[j%configurations_per_model] = (best_model_vmee, best_model_vloss, training_stats, model)
                else:
                    if configuration_best_model[j%configurations_per_model][0] >= best_model_vmee and configuration_best_model[j%configurations_per_model][1] >= best_model_vloss:
                        configuration_best_model[j%configurations_per_model] = (best_model_vmee, best_model_vloss, training_stats, model)

            structures_best_configurations.append(configuration_best_model)
        
        for i in range(len(structures_best_configurations)):
            print("(GS - MLCUP) - Structure", i)
            for j in range(len(structures_best_configurations[i])):
                print(f"(GS - MLCUP) - Configuration{j} : {structures_best_configurations[i][j][:-2]}")

        # evaluate models to find best
        best_model_info = None
        current_best_score = math.inf
        models_with_scores = []
        for i in range(len(structures_best_configurations)):
            # i-th model structure
            scores = []
            stats = []
            params = []
            models = []
            confidx = []
            for j in range(len(structures_best_configurations[i])):
                scores.append(self._compute_model_score(structures_best_configurations[i][j]))
                stats.append(structures_best_configurations[i][j][-2])
                params.append(self._get_model_parameters(j,len(structures_best_configurations[i])))
                models.append(structures_best_configurations[i][j][-1])
                confidx.append(j)
                models_with_scores.append((scores[-1], models[-1], stats[-1]))

            zipped_triples = sorted(zip(stats, scores, params, confidx, models), key = lambda x : x[1]) # sort everything by increasing score
            max_len = min(len(zipped_triples), 8) # to only get top best results for visualization sake
            stats   =            [x for x,_,_,_,_ in zipped_triples[:max_len]]
            scores  =            [x for _,x,_,_,_ in zipped_triples[:max_len]]
            params  =            [x for _,_,x,_,_ in zipped_triples[:max_len]]
            confidx =            [x for _,_,_,x,_ in zipped_triples[:max_len]]
            models  =            [x for _,_,_,_,x in zipped_triples[:max_len]]
            if zipped_triples[0][1] < current_best_score:
                best_model_info = (zipped_triples[0][-1], params[0], stats[0], self.models_layers[i])
                current_best_score = zipped_triples[0][1]

            print(f"(GS - MLCUP) - Model {i} evalutation")
            for j in range(max_len):
                print(f"(GS - MLCUP) - Configuration {confidx[j]}, score : {scores[j]}, best_validation_mee : {stats[j][-1][1]}, params:{params[j]}")

            Plot._plot_train_stats(stats,title=f"Model {i}", epochs=[x['epoch'] for x in params], block=(i==len(structures_best_configurations)-1), classification=False)

        if best_model_info is None:
            raise SystemError("No model was worth to be evaluated ( all negative score )")
        
        # save best models for ensemble exec
        ensemble_dim = 5
        models_with_scores = sorted(models_with_scores, key = lambda x : x[0])
        for i in range(ensemble_dim):
            save_file = f"models/ensemble_models/cup_ensemble{i}"
            with open(save_file, 'wb') as f:
                pickle.dump({"model": models_with_scores[i][1], "layers": models_with_scores[i][1].layers, "stats": models_with_scores[i][2]}, f, protocol=pickle.HIGHEST_PROTOCOL)

        return best_model_info

    def _get_model_parameters(self, index, configurations_per_model):
        epoch_len = max(configurations_per_model // len(self.epoch), 1)
        epoch = self.epoch[index // epoch_len]
        index = index % epoch_len # shift inside single epoch
        batch_size_len = max(epoch_len // len(self.batch_size), 1)
        batch_size = self.batch_size[index // batch_size_len]
        index = index % batch_size_len # shift inside single batch_size
        lr_decay_len = max(batch_size_len // len(self.lr_decay), 1)
        lr_decay = self.lr_decay[index // lr_decay_len]
        index = index % lr_decay_len # shift inside single lr_decay
        eta_len = max(lr_decay_len // len(self.eta), 1)
        eta = self.eta[index // eta_len]
        index = index % eta_len # shift inside single eta
        alpha_len = max(eta_len // len(self.alpha), 1)
        alpha = self.alpha[index // alpha_len]
        index = index % alpha_len # shift inside single alpha
        alpha_len = max(alpha_len // len(self.alpha), 1)
        _lambda = self._lambda[index // alpha_len]

        return {'epoch':epoch, 'batch_size':batch_size, 'lr_decay':lr_decay, 'eta':eta, 'alpha':alpha, '_lambda':_lambda}

    def get_model_perturbations(self, model_conf, model_architecture, weight_range=None):
        layers=[model_architecture]
        weight_range=[(-0.69, 0.69)] if weight_range is None else [weight_range]
        batch_size=[model_conf['batch_size']]
        epoch=[model_conf['epoch']]
        lr_decay=[model_conf['lr_decay']]

        eta=[]
        alpha=[]
        _lambda=[]

        eta_perturb = model_conf['eta'] / 2
        alpha_perturb = 0.1
        lambda_perturb = model_conf['_lambda'] / 2

        # keep original model_conf
        eta.append(model_conf['eta'])
        alpha.append(model_conf['alpha'])
        _lambda.append(model_conf['_lambda'])

        # perturb to create new random model_confs
        for i in range(4):
            eta.append(np.random.uniform(model_conf['eta'] - eta_perturb, model_conf['eta'] + eta_perturb))
        for i in range(2):
            alpha.append(max(min(np.random.uniform(model_conf['alpha'] - alpha_perturb, model_conf['alpha'] + alpha_perturb), 0.99), 0))
        if model_conf['_lambda'] != 0:
            for i in range(2):
                _lambda.append(np.random.uniform(model_conf['_lambda'] - lambda_perturb, model_conf['_lambda'] + lambda_perturb))
        return layers, weight_range, batch_size, epoch, lr_decay, eta, alpha, _lambda