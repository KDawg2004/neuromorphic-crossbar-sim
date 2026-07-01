class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x, dt):
        if not self.layers:
            raise RuntimeError("Neural network contains no layers.")
        for layer in self.layers:
            x = layer.forward(x, dt)

        return x