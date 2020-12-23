import numpy as np # more important than "#include <stdio.h>"

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



