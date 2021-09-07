import numpy as np

# output that could be given from the output layer of the nn
softmax_output = [0.7, 0.1, 0.2]
# correct class labelled as the 0th class
target_output = [1, 0, 0]

# calculate loss function using categorical cross entropy
# essentially negative log of the softmax output of the labelled class
loss = -(np.log(softmax_output[0]) * target_output[0] +
         np.log(softmax_output[1]) * target_output[1] +
         np.log(softmax_output[2]) * target_output[2])

print(loss)
