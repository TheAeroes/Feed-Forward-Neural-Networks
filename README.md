## L-layer Feed Forward ANN 

### Capabilities: 
 -  Initialization: Xavier and He et al.  
 -  Activation : ReLU, sigmoid, softmax
 -  Cost functions: Bernoulli and Multinoulli cross-entropy
 -  Optimization: Momentum and ADAM 
 -  Regularization: L2 and Dropout
 
 ### Dependencies:
 - numpy
 - matplotlib
 - math 

### Example
```
activation_list = ['ReLU','ReLU','softmax']   # define activations for each layer and output layer
layer_dims = [200,200,10]   # dimensions for each layer

# Running the model
"""
X -- input data, of shape (input size, number of examples) numpy array
Y -- true "label" vector  (1, number of examples) numpy array  where labels are 0 to num_classes-1
"""
parameters = ff_nn.model(X_train, Y_train, layer_dims, activation_list, optimizer ='adam', learning_rate = 0.0007, 
                      mini_batch_size =100,beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 100 , 
                      print_cost = True, print_every = 1, lambd = 0.001, keep_prob = 0.8)

# Calculating Accuracy 
acc_train = ff_nn.performance_metric(Y_train,X_train,parameters,activation_list)
acc_test =  ff_nn.performance_metric(Y_test,X_test,parameters,activation_list)


```
 
