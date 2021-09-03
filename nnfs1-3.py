import numpy as np

# neural network with 2 layers, 1st layer has 4 neurons, second has 3
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights1 = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases1 = [2, 3, 0.6]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, 0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [1, 2, 0.5]

# dot product the inputs with the transpose of the weights, add the bias for each layer
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer1_outputs)
print(layer2_outputs)

print("test")


