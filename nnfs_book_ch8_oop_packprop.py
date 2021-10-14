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
plt.show()

np.random.seed(0)


# same as nnfs5
class LayerDense:
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
        self.inputs = inputs

    # takes in derivative of last layer
    def backward(self, dvalues):
        # chain rule for weights is inputs of the layer (transposed so we can dot) * dvalues
        self.dweights = np.dot(self.inputs.T, dvalues)
        # chain rule for inputs is the dvalues . weights
        self.dinputs = np.dot(dvalues, self.weights.T)
        # derivative of biases is just the sum of the previous layer derivatives
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)


# ReLU activation function if x <=0, return 0 else return x
class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        # remember the input values for when we backprop
        self.inputs = inputs

    # take in the derivatives from the layer infront
    def backwards(self, dvalues):
        # make a copy of the variable so we dont edit the original
        self.dinputs = dvalues.copy()
        # derivative of relu is 1 for x >= 1 and 0 for x < 0, so dvalues * drelu:
        self.dinputs[self.inputs <= 0] = 0


# softmax activation function for output layer
class ActivationSoftmax:
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
class LossCategoricalCrossEntropy(Loss):
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

    # backprop
    def backward(self, dvalues, y_true):
        # find number of samples
        samples = len(dvalues)
        # find number of labels
        labels = len(dvalues[0])

        # if labels are sparse (0,2,3,1), convert to one hot
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # calculate gradient
        self.dinputs = -y_true / dvalues
        # normalise gradient
        self.dinputs = self.dinputs / samples


# finding accuracy using argmax inheriting from loss
class AccuracyArgMax(Loss):
    def forward(y_prediction, y_true):
        if len(y_true.shape) == 2:
            y_prediction = np.argmax(y_true == 1, axis=1)
        predictions = np.argmax(y_prediction, axis=1)
        y_accuracy = np.mean(predictions == y_true)
        return y_accuracy


# initialise data / obj classes
# X, y = create_data(100, 3)
# 1st layer has to have 2 inputs, since data only 2 inputs
dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()
dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

# create nn
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# print first 5 outputs
print(activation2.output[:5])

# calculate loss
loss_function = LossCategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)
print(loss)

accuracy_function = AccuracyArgMax()
accuracy = accuracy_function.calculate(activation2.output, y)
print(accuracy)
