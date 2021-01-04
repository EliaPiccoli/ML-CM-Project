import numpy as np # more important than "#include <stdio.h>"
import utils.get_dataset as dt
import copy

from utils.layer import Layer
from utils.model import Model
from utils.plot import Plot
from tqdm import trange

# IDEA:
# vogliamo testare modelli con vari parametri (momentum, batch_size, diverse strutture di rete, lr/decay, non mi viene in mente altro) 
# vogliamo farlo per ottenere dei risultati che ci permettano di capire quale modello performa meglio
# di solito per capire come si comporta una cosa andiamo a vedere il comportamento medio e potremmo farlo anche qui, magari facendo fare 
# 20 diversi training ad ognuno dei modelli. Avremmo comunque del bias dovuto al fatto che siamo MOLTO dipendenti dalle inizializzazioni
# dei weight e dei bias. Come risolvere? se vogliamo fare 20 train creiamo 20 configurazioni iniziali di weight e bias, faremo runnare
# tutti i modelli su quelle 20 configurazioni, così non ci possono essere storie a riguardo.

# ANDANDO + NEL DETTAGLIO IMPLEMENTATIVO:
# pensavo come risultato della grid search un qualcosa del genere, assumendo di avere per esempio 3 diverse configurazioni da provare

# [(best_model_1, avg_test_accuracy1),(best_model_2, avg_test_accuracy2),(best_model_3, avg_test_accuracy3)]

# dove i vari avg_test_accuracy sono, ipotizzando 20 training per ogni model: (test_acc1 + ... + test_acc20) / 20
# best_model invece deve essere l oggetto model più performante di tutti e 20 i training, pensavo di trovarlo tramite  2 livelli di ricerca:
# - all interno del training la funzione validation si deve occupare (basandosi sui valori di val_acc e val_loss) di tenere i pesi ed i bias 
# che hanno dato il miglior risultato (che ne so, un paio di campi del tipo self.best_weight e self.best_bias per immagazzinare e 
# self.best_val_acc e self.best_val_loss per confrontare)
# a questo punto, finito il training abbiamo in mano il modello tecnicamente migliore (ci bastano i pesi ed i bias per l inferenza)
# dunque usiamo quello per vedere la test accuracy.
# - quindi per OGNI training (di una particolare configurazione) avremo il modello più performante e la relativa test accuracy, a questo punto possiamo 
# trovare il modello più performante di tutta la configurazione basandoci innanzitutto sulla test accuracy, se poi abbiamo casi di parità possiamo 
# dare un occhio alla best_val_acc e best_val_loss, che saranno la val_acc e val_loss del modello performante che abbiamo in mano

# Selezionando il best_model e facendo la media tra le test_accuracy che abbiamo generato nei vari training possiamo costruire la tupla 
# di cui parlavo sopra

# Ripetere per ogni configurazione

# Alla fine avremo una lista con l "ottimo" per ogni configurazione, così da permetterci di farci tutti i grafici del caso e di sceglierci
# quello che ci piace di più (ovviamente anche sta scelta è automatizzabile)

# Cosa è gridsearchable?
# eta, alpha, (_lambda,) batch_size, layer, #nodes, lr_decay, weight_range, epoch

class GridSearch:
    def __init__(self):
        self.eta = [0.01]
        self.alpha = [0]
        self._lambda = [0]
        self.batch_size = [1]
        self.models_layers = [] # [[Layers#1], [Layers#2], ...]
        self.lr_decay = [1e-5]
        self.epoch = [300]
        self.weight_range = [(-0.69, 0.69)]

    def _set_parameters(self, **parameters):
        """
            parameters:
                eta : list, 
                alpha : list, 
                _lambda : list, 
                batch_size : list, 
                layers : list of list of Layers,
                lr_decay : list,
                epoch : list,
                weight_range : list of tuple (Lower, Upper)
        """
        if "eta" in parameters:
            self.eta = parameters["eta"]                               # [0.1, 0.0001]
        if "alpha" in parameters:
            self.alpha = parameters["alpha"]                           # [0.6, 0.98]
        if "_lambda" in parameters:
            self._lambda = parameters["_lambda"]                       # [1e-3, 1e-5]
        if "batch_size" in parameters:
            self.batch_size = parameters["batch_size"]                 # [len(inp), 1]
        if "layers" in parameters:
            self.models_layers = parameters["layers"]               
        if "lr_decay" in parameters:
            self.lr_decay = parameters["lr_decay"]                     # [1e-5 1e-6]
        if "epoch" in parameters:
            self.epoch = parameters["epoch"]                           # [100 1000]
        if "weight_range" in parameters:
            self.weight_range = parameters["weight_range"]
    
    def _compute_model_score(self, model_infos):
        # model_infos : (avg_test_acc, (best_model, test_acc_bm, training_bm[(a, va, l, vl)]))
        score = 0
        # test accuracy
        score += 1000*model_infos[0]
        # validation loss smooth and training loss smooth (val has more weight)
        val_loss = []
        train_loss = []
        threashold = 5e-3
        for epoch in model_infos[1][2]:
            val_loss.append(epoch[3])
            train_loss.append(epoch[2])
        not_decrease_times = 0
        for i in range(len(val_loss)-1):
            if val_loss[i+1] - val_loss[i] > threashold:
                not_decrease_times += 1
        score += -not_decrease_times*10

        not_decrease_times = 0
        for i in range(len(train_loss)-1):
            if train_loss[i+1] - train_loss[i] > threashold:
                not_decrease_times += 1
        score += -not_decrease_times*2

        return score

    
    # TODO: add loss function name for model (not hyperparameter) (atm set it as defualt)
    def _run(self, train, train_label, validation, validation_label, test, test_label, familyofmodelsperconfiguration=5):
        print("I am not fast, sorry")
        print("Maybe in future i will use all your cores")
        print("Maybe the GPU, who knows")

        print("Generating weights")
        weights_per_configuration = []         # confs: [ weight_range_inits:[ weight_inits: [particular weight matrix]]]
        for configuration in self.models_layers:
            dimensions = [] # [(in, out), (in, out)] for each layer
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
        # for i in range(len(weights_per_configuration)):
        #     print("Conf: ", i)
        #     for j in range(len(weights_per_configuration[i])):
        #         print("Init: ", j)
        #         for k in range(len(weights_per_configuration[i][j])):
        #             print("Layer: ", k)
        #             print("Weights: ", weights_per_configuration[i][j][k])
        #     print("---")

        # will use range(max(len, 1)) so if any value for whatever list was not provided it will iterate just one time using the default value
        # max is useless but is more clear what happens if the hyperparameter was not considered
        # if missing value the class initialize all the lists to the default value
        # list of tuple (epoch, batch, decay, compiled_model)
        print("Generating models")
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
                                        # print("eta: {} - aplha: {} - lambda: {}".format(self.eta[eta_index], self.alpha[alpha_index], self._lambda[lambda_index]))
                                        # print("epoch: {} - batch: {} - decay: {}".format(self.epoch[epoch_index], self.batch_size[batch_size_index], self.lr_decay[decay_index]))
                                        # initialize model
                                        model = Model()
                                        weights_matrix = []
                                        # print("model ", hex(id(model)))
                                        for k in range(len(weights_per_configuration[i][j])):
                                            model_layer = self.models_layers[counter//(len(self.weight_range)*familyofmodelsperconfiguration)][k]
                                            layer = Layer(model_layer.nodes, model_layer.activation_function_type, _input=model_layer.input)
                                            # print("layer ", hex(id(layer)))
                                            model._add_layer(layer)
                                            weights_copy = []
                                            for node_weights in weights_per_configuration[i][j][k]:
                                                weights_copy.append([])
                                                for weight in node_weights:
                                                    weights_copy[-1].append(weight)
                                            weights_matrix.append(weights_copy)
                                            # weights_matrix.append(weights_per_configuration[i][j][k]) <- fucking pointers, all models shared the same weight matrix
                                        model._compile(eta=self.eta[eta_index], alpha=self.alpha[alpha_index], _lambda=self._lambda[lambda_index], weight_matrix=weights_matrix)
                                        # print(model)
                                        models_configurations.append((self.epoch[epoch_index], self.batch_size[batch_size_index], self.lr_decay[decay_index], model))
                counter += 1
        # print(len(self.epoch)*len(self.batch_size)*len(self.lr_decay)*len(self.eta)*len(self.alpha)*len(self._lambda)*len(self.weight_range)*familyofmodelsperconfiguration)
        print(f"Generated {len(models_configurations)} diffent models.")
        print("Starting Training")
        # TODO: parallelize
        models_per_structure = len(models_configurations) // len(self.models_layers)
        configurations_per_model = len(self.epoch)*len(self.batch_size)*len(self.lr_decay)*len(self.eta)*len(self.alpha)*len(self._lambda)

        structures_best_configurations = []
        for i in range(len(self.models_layers)):
            print("Model ", i)
            configuration_test = [0]*configurations_per_model
            configuration_best_model = [None]*configurations_per_model
            for j in range(models_per_structure):
                epoch, batch, decay, cur_model = models_configurations[i*models_per_structure + j]
                print("Start training ", i*models_per_structure + j)
                training_stats = cur_model._train(train, train_label, validation, validation_label, batch_size=batch, epoch=epoch, decay=decay)
                print("Start test ", i*models_per_structure + j)
                test_accuracy = cur_model._infer(ohe_test, test_exp)
                configuration_test[j%configurations_per_model] += test_accuracy/(len(self.weight_range)*familyofmodelsperconfiguration)
                if configuration_best_model[j%configurations_per_model] is None:
                    configuration_best_model[j%configurations_per_model] = (cur_model.best_model, test_accuracy, training_stats)
                else:
                    if configuration_best_model[j%configurations_per_model][1] < test_accuracy:
                        configuration_best_model[j%configurations_per_model] = (cur_model.best_model, test_accuracy, training_stats)
                    elif configuration_best_model[j%configurations_per_model][1] == test_accuracy:
                        if configuration_best_model[j%configurations_per_model][0].validation_accuracy < cur_model.best_model.validation_accuracy:
                            configuration_best_model[j%configurations_per_model] = (cur_model.best_model, test_accuracy, training_stats)
                        elif configuration_best_model[j%configurations_per_model][0].validation_accuracy == cur_model.best_model.validation_accuracy:
                            if configuration_best_model[j%configurations_per_model][0].validation_loss > cur_model.best_model.validation_loss:
                                configuration_best_model[j%configurations_per_model] = (cur_model.best_model, test_accuracy, training_stats)
            configurations_results = []
            for k in range(configurations_per_model):
                configurations_results.append((configuration_test[k], configuration_best_model[k]))
            structures_best_configurations.append(configurations_results)
        
        for i in range(len(structures_best_configurations)):
            print("Structure", i)
            for j in range(len(structures_best_configurations[i])):
                print("Configuration", j)
                print(structures_best_configurations[i][j][0], structures_best_configurations[i][j][1][:-1])

        # evaluate models to find best
        for i in range(len(structures_best_configurations)):
            # i-th model structure
            scores = []
            stats = []
            params = []
            for j in range(len(structures_best_configurations[i])):
                scores.append(self._compute_model_score(structures_best_configurations[i][j]))
                stats.append(structures_best_configurations[i][j][1][-1])
                params.append(self._get_model_parameters(j,len(structures_best_configurations[i])))

            zipped_triples = sorted(zip(stats, scores, params), key = lambda x : x[1], reverse = True) # sort everything by decreasing score
            max_len = min(len(zipped_triples), 8) # to only get top best results for visualization sake
            stats  = [x for x,_,_ in zipped_triples[:max_len]]
            scores = [x for _,x,_ in zipped_triples[:max_len]]
            params = [x for _,_,x in zipped_triples[:max_len]]

            for j in range(max_len):
                print(f"Configuration {j}, score : {scores[j]}, params:{params[j]}")

            Plot._plot_train_stats(stats,title=f"Model {i}", epochs=[x['epoch'] for x in params], block=(i==len(structures_best_configurations)-1))

    def _get_model_parameters(self, index, configurations_per_model):
        # ALL THIS IS FOR COMPREHENSION ONLY, TUTTO RIDUCIBILE AD UN CICLO VOLENDO, 5-6 RIGHE MAX TRANQUI EP NO RABIA
        # len(self.epoch)*len(self.batch_size)*len(self.lr_decay)*len(self.eta)*len(self.alpha)*len(self._lambda)
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


                                 
if __name__ == "__main__":
    gs = GridSearch()
    train, validation, train_labels, validation_labels = dt._get_train_validation_data(1, split=0.25)
    models = [[Layer(4, "tanh", _input=(17,)), Layer(1, "tanh")],[Layer(4, "tanh", _input=(17,)), Layer(4, "tanh"), Layer(1, "tanh")]]
    gs._set_parameters(layers=models, weight_range=[(-0.69, 0.69)], eta=[0.1, 0.01, 0.001, 0.0001], alpha=[0.6,0.85, 0.98], batch_size=[1,16,32,len(train_labels)], epoch=[300,500])
    #gs._set_parameters(layers=models, weight_range=[(-0.69, 0.69)], eta=[0.01,0.0001], alpha=[0.85,0.98], batch_size=[16,len(train_labels)], epoch=[300,500])
    ohe_inp = [dt._get_one_hot_encoding(i) for i in train]
    ohe_val = [dt._get_one_hot_encoding(i) for i in validation]
    train_exp = [[elem] for elem in train_labels]
    validation_exp = [[elem] for elem in validation_labels]
    test, test_labels = dt._get_test_data(1)
    ohe_test = [dt._get_one_hot_encoding(i) for i in test]
    test_exp = [[elem] for elem in test_labels]
    gs._run(ohe_inp, train_exp, ohe_val, validation_exp, ohe_test, test_exp, familyofmodelsperconfiguration=5)




                

