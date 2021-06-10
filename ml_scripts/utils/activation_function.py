from .relu import Relu
from .sigmoid import Sigmoid
from .tanh import Tanh
from .leaky_relu import LeakyRelu
from .linear import Linear

AF = {
    "sigmoid" : Sigmoid(),
    "relu" : Relu(),
    "leaky_relu" : LeakyRelu(),
    "tanh" : Tanh(),
    "linear" : Linear()
}