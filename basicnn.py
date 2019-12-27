import numpy as np
import math

def sigmoid(x):
    """
    Calculates sigmoid(x)
    """"
    return 1 / (1 + math.exp(-x))

class NeuralNetwork:
    def __init__(self, x, y):
        self.input  = x
        self.w1 = np.random.rand(self.input.shape[1], 4) #weight 1
        self.w2 = np.random.rand(4, 1) #weight 2
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        #assuming bias = 0
        self.layer1 = sigmoid(np.dot(self.input, self.w1))
        self.output = sigmoid(np.dot(self.layer1, self.w2))


