import numpy as np

np.random.seed(0)

# training dataset (keep small otherwise values explode as they pass through nn) 0 < values < 0.1
# initialise biases as 0 (as long as weights arent 0)

# 4 sample data per input, 3 batches of sample data
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


# 2 hidden layers
class Layer_Dense:
    # initialise the class
    def __init__(self, n_inputs, n_neurons):
        # Gaussian distribution rounded at 0, in a matrix dimensions n_inputs by n_neurons
        # 0.1 to keep weights small
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # create 1 x n_neurons matrix of zeroes for biases (pass into as tuples)
        self.biases = np.zeros((1, n_neurons))

    # calculates the output of the layer dot product the inputs and weights, add biases
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# initialise the 2 layers
layer1 = Layer_Dense(4, 5)
# output from layer 1 has to be same shape as input to layer 2
layer2 = Layer_Dense(5, 6)

# pass the test data forward through layer 1
layer1.forward(X)
# pass layer 1 output through layer 2
layer2.forward(layer1.output)

# print values of each layer
print(layer1.output)
print(layer2.output)
