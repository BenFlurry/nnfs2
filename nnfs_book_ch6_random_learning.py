import numpy as np
from nnfs.datasets import vertical_data, spiral_data
import nnfs
# IGNORE
# create dataset
from matplotlib import pyplot as plt

nnfs.init()

X, y = vertical_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.plot()
# plt.show()

np.random.seed(0)

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


class Loss:
    # output from model, y -> intended target values
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


# loss function inheriting from loss
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_prediction, y_true):
        # find number of samples
        samples = len(y_prediction)
        # clip values so not to infinity
        y_prediction_clipped = np.clip(y_prediction, 1e-7, 1 - 1e-7)
        # 1d array of target value
        if len(y_true.shape) == 1:
            correct_confidences = y_prediction_clipped[range(samples), y_true]
        # 1 hot encoded vectors
        elif len(y_true.shape) == 2:
            # multiple the hot coded vector and the prediction vector and sum across each sample
            correct_confidences = np.sum(y_prediction * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# finding accuracy using argmax inheriting from loss
class Accuracy_ArgMax(Loss):
    def forward(self, y_prediction, y_true):
        if len(y_true.shape) == 2:
            y_prediction = np.argmax(y_true == 1, axis=1)
        predictions = np.argmax(y_prediction, axis=1)
        y_accuracy = np.mean(predictions == y_true)
        return y_accuracy


# initialise data / obj classes
# X, y = create_data(100, 3)
# 1st layer has to have 2 inputs, since data only 2 inputs
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossEntropy()
# create nn

# Helper variables
lowest_loss = 9999999
highest_accuracy = 0
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):
    # create new set of weights for the iteration
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    # forward pass of the training data
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # perform a forward pass through teh activation function
    # it takes the output of the second dense layer and returns the loss
    loss = loss_function.calculate(activation2.output, y)

    # calculate accuracy from output of activation 2 and targets
    # calculate along the first axis
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    # if loss is smaller, print and save weights and biases
    if loss < lowest_loss:
        print(f'New best weights in iteration: {iteration}, loss : {loss}, accuracy : {accuracy}')
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss

    # revert weights and biases if not the best
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

    if accuracy > highest_accuracy:
        highest_accuracy = accuracy

print(lowest_loss, highest_accuracy)
