from matplotlib.pyplot import axis
import numpy as np

class SoftMax:

    def __init__(self):
        pass

    def forward(self, input_tensor):
        self.input_tensor = input_tensor.copy() # - input_tensor.max()
        self.input_tensor = np.exp(self.input_tensor - np.max(self.input_tensor))
        # the_sum = np.exp(input_tensor).sum()
        self.output_tensor = self.input_tensor / self.input_tensor.sum(axis=1)
        # print('input_tensor = ', self.input_tensor.shape, 'output_tensor = ', self.output_tensor.shape)
        return self.output_tensor

    def backward(self, error_tensor):
        locsum = error_tensor * self.output_tensor.sum(axis=0)
        locsum = locsum.sum()
        # print('output_tensor = ', self.output_tensor.shape, 'error_tensor = ', error_tensor.shape)
        return self.output_tensor * (error_tensor - locsum)