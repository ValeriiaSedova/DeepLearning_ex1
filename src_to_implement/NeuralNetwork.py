import copy

class NeuralNetwork:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        activation = self.input_tensor
        for layer in self.layers:
            activation = layer.forward(activation)

        return self.loss_layer.forward(activation, self.label_tensor)

    def backward(self):
        # input_tensor, label_tensor = self.data_layer.next()
        back_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            back_tensor = layer.backward(back_tensor)

        return back_tensor
        
    def append_trainable_layer(self, layer):
        loc_optimizer = copy.deepcopy(self.optimizer)
        layer.optimizer = loc_optimizer
        self.layers.append(layer)

    def train(self, iterations):
        for iter in range(iterations):
            loss = self.forward()
            error_tensor = self.backward()
            self.loss.append(loss)

    def test(self, input_tensor):
        prediction = input_tensor
        for layer in self.layers:
            prediction = layer.forward(prediction)
        return prediction

    