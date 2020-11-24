import copy

class NeuralNetwork:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        activation = input_tensor
        for layer in self.layers:
            activation = layer.forward(activation)

        return activation

    def backward(self):
        input_tensor, label_tensor = self.data_layer.next()
        back_tensor = self.loss_layer(label_tensor)
        for layer in reversed(self.layers):
            back_tensor = layer.backward(back_tensor)

        return back_tensor
        
    def append_trainable_layer(self, layer):
        loc_optimizer = copy.deepcopy(self.optimizer)
        layer.optimizer = loc_optimizer
        self.layers.append(layer)

    def train(self, iterations):
        for iter in range(iterations):
            input_tensor, label_tensor = self.data_layer.next()
            prediction = self.forward(input_tensor)
            error_tensor = self.backward(label_tensor)
            self.loss.append(error_tensor.sum())

    def test(self, input_tensor):
        prediction = self.forward(input_tensor)
        return prediction

    