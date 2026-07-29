import numpy as np
inputs = [[1,2,3,2.5],
          [2.0,5,-1,2],
          [-1.5,2.7,3.3,-0.8]]

weights = [[.2,.8,-.5,1.0],
           [.5,-.91,.26,-.5],
           [-.26,-.27,.17,.87]]
bias =[2 ,3,.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

layer1_output= np.dot(inputs, np.array(weights).T)+bias
layer2_output = np.dot(layer1_output, np.array(weights2).T)+biases2
print(layer2_output)

#print();

#print(np.array(weights));