import numpy as np

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

# softmax activation function
# exponentiate all values to get rid of negative values
exp_values = np.exp(layer_outputs)

# normalisation distribution - exp of each inp / sum of exp of each set of inp
# axis=1 -> adds each set of inputs, not whole matrix
# keepdims=True -> keeps dimensions of added inputs the same (rather than being nx1, its 1xn)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(norm_values)

'''
exponentials get very big very quick, so we can do the for each set of inputs:
inputs - max(inputs) so the largest of each set of inputs is 0, therefore we don't get overflow
after exponentiation and normalisation, the outputted values are the same using this method 
'''

