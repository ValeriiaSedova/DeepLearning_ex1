import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        pass

    def forward(self, input_tensor, label_tensor):
       self.loss = - np.ln(input_tensor + np.finfo(float).eps).sum()
       return self.loss

    def backward(self, label_tensor):
        error_tensor = - 1 / label_tensor
        return error_tensor 