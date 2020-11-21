import numpy as np

class FullyConnected:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        #maybe +1 somewhere
        self.weights = np.random.random([self.input_size,self.output_size])

    def forward(self, input_tensor):
        input_tensor = input_tensor.T
        print('FORWARD:',self.weights.T.shape, input_tensor.shape)
        return np.dot(self.weights.T, input_tensor).T

    def backward(self, error_tensor):
        self.error_tensor = error_tensor.T
        self.gradient_weights = np.dot(self.weights, self.error_tensor).T
        return  self.gradient_weights #maybe save it as an attribute later on

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    