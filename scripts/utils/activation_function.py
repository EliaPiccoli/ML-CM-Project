from .relu import Relu
from .sigmoid import Sigmoid
from .softmax import Softmax
from .tanh import Tanh
from .leaky_relu import LeakyRelu

AF = {
    "sigmoid" : Sigmoid(),
    "relu" : Relu(),
    "leaky_relu" : LeakyRelu(),
    "softmax" : Softmax(),
    "tanh" : Tanh()
}