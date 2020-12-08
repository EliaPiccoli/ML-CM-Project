from .relu import Relu
from .sigmoid import Sigmoid
from .softmax import Softmax

AF = {
    "sigmoid" : Sigmoid(),
    "relu" : Relu(),
    "softmax" : Softmax()
}