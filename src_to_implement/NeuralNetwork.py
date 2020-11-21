class NeuralNetwork:

    def __init__(self, optimizer, loss, layers, data_layer, loss_layer):
        self.optimizer = optimizer
        self.loss = loss
        self.layers = layers
        self.data_layer = data_layer
        self.loss_layer = loss_layer

    def forward(self):
        for input_tensor, label_tensor in self.data_layer:
            pass
        return None

    