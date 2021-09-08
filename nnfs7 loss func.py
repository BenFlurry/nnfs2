import numpy as np

# output that could be given from the output layer of the nn
softmax_output1 = [0.7, 0.1, 0.2]
# correct class labelled as the 0th class
target_output1 = [1, 0, 0]

# calculate loss function using categorical cross entropy
# essentially negative log of the softmax output of the labelled class
loss1 = -(np.log(softmax_output1[0]) * target_output1[0] +
          np.log(softmax_output1[1]) * target_output1[1] +
          np.log(softmax_output1[2]) * target_output1[2])

print(loss1)

# function represented using numpy arrays, so we can calculate loss for batches
softmax_output2 = np.array([[0.1, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

# going down the batches, target index is 0, then 1, then 1
class_targets2 = [0, 1, 1]

# softmax function returning vector of each loss per batch
# issue with log(0) = infinity -> use np.clip
loss2 = -np.log(softmax_output2[range(len(softmax_output2)), class_targets2])
# find average of the losses from each batch
batch_loss = np.mean(loss2)
