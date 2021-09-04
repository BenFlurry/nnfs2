import numpy as np
import matplotlib.pyplot as plt


# IGNORE
# create dataset
def create_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


# same as nnfs5
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


# ReLU activation function if x <=0, return 0 else return x
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# softmax activation function for output layer
class Activation_Softmax:
    def forward(self, inputs):
        # explained in nnfs6 softmax func.py
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = prob_values


# initialise data / obj classes
X, y = create_data(100, 3)
# 1st layer has to have 2 inputs, since data only 2 inputs
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# create nn
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

# print first 5 outputs
print(activation2.output[:5])

''' 
since initialisation was random with random data, we get a near perfect distribution of 1/3 for each value
'''
