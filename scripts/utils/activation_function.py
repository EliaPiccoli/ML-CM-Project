from .relu import Relu
from .sigmoid import Sigmoid
from .softmax import Softmax
from .tanh import Tanh

AF = {
    "sigmoid" : Sigmoid(),
    "relu" : Relu(),
    "softmax" : Softmax(),
    "tanh" : Tanh()
}