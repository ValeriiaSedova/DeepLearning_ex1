import numpy as np

class FullyConnected:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.random([self.input_size + 1, self.output_size])
        self._optimizer = None

    def forward(self, input_tensor):
        height, width = input_tensor.shape
        self.input_tensor = np.concatenate((input_tensor, np.ones([height, 1])), axis = 1)
        y_hat = np.dot(self.input_tensor, self.weights)
        return y_hat 

    def backward(self, error_tensor):
        error_tensor_new = np.dot( error_tensor, self.weights[:-1,:].T)
        if self._optimizer != None: 
            self.gradient = np.dot(self.input_tensor.T, error_tensor)
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient)        
        return error_tensor_new

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    