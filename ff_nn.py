# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 08:54:16 2017

@author: Eyas
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.sparse as sp

#%%
def split_data(per_val,n_points,per_train = [] ,seed = 0):
    
    idx_p = np.arange(n_points)
    
    np.random.seed(seed)
    n_val = int(per_val*n_points)
    idx_val  = np.random.choice(idx_p, size= n_val, replace = False)

    if not per_train:    
        idx_train = np.setdiff1d(idx_p,idx_val)
        idx = {'training':idx_train,'validation':idx_val}
        
    else:
        n_train = int(per_train*n_points)
        idx_train = np.random.choice(np.setdiff1d(idx_p,idx_val), size= n_train, replace = False)
        idx_test = np.setdiff1d(idx_p,np.concatenate((idx_train,idx_val)))
        idx = {'training':idx_train,'validation':idx_val,'testing':idx_test}

    
    return idx
#%%
def initialize_parameters(layer_dims, init_type = 'random' , scale_f = 0.01, seed = 0 , init_parameters = []):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(seed)
    parameters = {}
    L = len(layer_dims)   # number of layers in the network
    
    if init_type  == 'random':
        sF = lambda x: scale_f
    elif init_type == 'He': # He et al. 2015 initialization  (good for ReLU)
        sF = lambda x: np.sqrt(2/x)
    elif init_type == 'Xavier': # Xavier initialization
        sF = lambda x: np.sqrt(1/x)
    elif init_type == 'Specific': # initialize with specific parameters
        parameters = init_parameters
   
    if init_type !='Specific':    
        for l in range(1, L):
        
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * sF(layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
     
    return parameters
#%%
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    if sp.issparse(A):
        Z = A.T.dot(W.T).T +b   
    else:
        Z = np.dot(W,A)+b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache
#%%
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1/(1+np.exp(-z));
    cache = z
    
    return s , cache
#%%
def softmax(z):
    """
    Compute the softmax of z

    Arguments:
    z --  numpy array of any size.

    Return:
    s -- softmax(z)
    """
    log_s = z  - log_sum_exp(z)
    s  = np.exp(log_s)
    #max_z = np.max(z,0)
    #s = np.exp(z-max_z)/np.sum(np.exp(z-max_z),0);
    cache = z
    
    return s , cache
#%%
def log_sum_exp(Z):
    
    max_z = np.max(Z,0)
    
    log_s_e = np.log(np.sum(np.exp(Z -max_z),0)) + max_z
    
    return log_s_e

#%%
def ReLU(z):
    """
    Compute the ReLu of z
    
    Arguments:
    z --  numpy array of any size.

    Return:
    s -- ReLu(z)
    """
   # s = np.copy(z)
    #s[s<=0] = 0
    s = np.maximum(0,z);
    cache = z
    return s, cache  
#%%    
def linear_activation_forward(A_prev, W, b, activation, keep_prob =1):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)

    elif activation == "ReLU":
        A, activation_cache = ReLU(Z)
    
    elif activation == "softmax":
        A, activation_cache = softmax(Z)
 
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    
    if keep_prob < 1:
        D = np.random.rand(A.shape[0],A.shape[1])  # Step 1: initialize matrix D = np.random.rand(..., ...)
        D = (D<keep_prob)                          # Step 2: convert entries of D to 0 or 1 (using keep_prob as the threshold)
        A = A*D                                    # Step 3: shut down some neurons of A
        A = A/keep_prob                            # Step 4: scale the value of neurons that haven't been shut down
        cache = (linear_cache, activation_cache,D)
    else: 
        cache = (linear_cache, activation_cache)
        

    return A, cache
#%%
def forward_propagation(X, parameters, activation_list , keep_prob = 1):
    """
    Implement forward propagation 
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    np.random.seed(1)
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)] , parameters['b' + str(l)], activation_list[l-1], keep_prob)
        caches.append(cache)
            

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)] , parameters['b' + str(L)], activation_list[L-1],keep_prob =1)
    caches.append(cache)
    
    
  #  assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches
#%%
def compute_cost(AL, Y, cost_type, parameters = [], lambd = 0 , onehot = False, Y_start=0):
    """
    Implement the cost function

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """  
    K, m  = AL.shape
    
    L2_regularization_cost = 0;
    
    if lambd>0:
        L = len(parameters) // 2  # number of layers in the neural network
        for l in range(1, L): 
            L2_regularization_cost += np.sum(np.square(parameters['W' + str(l)]))
        L2_regularization_cost *= lambd/(2*m)

    
    # Compute loss from aL and y.
    if cost_type == "sigmoid":
        cost = -float(np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))/m)
        #cost = -np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))/m
    elif cost_type =='softmax':
        cost = -np.sum(np.log(AL[(Y-Y_start),range(m)]))/m
        
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    cost += L2_regularization_cost
    
    assert(cost.shape == ())
    
    return cost
#%%
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    if sp.issparse(A_prev):
        dW = A_prev.dot(dZ.T).T/m
    else:
        dW = np.dot(dZ,A_prev.T)/m
    
    db = np.sum(dZ,axis=1,keepdims= True)/m
    dA_prev = np.dot(W.T,dZ)

    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db
#%%
def relu_backward(dA, cache):
    
    #dZ = dA*(activation_cache>0)
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ
#%%
def sigmoid_backward(dA, cache):
    
    #dZ = dA*activation_cache*(1- activation_cache)
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

#%%
def softmax_backward(dAL,cache): 
    Z = cache 
    SF,_ = softmax(Z)
    dZ = SF - dAL
    
    return dZ
#%%
def linear_activation_backward(dA, cache, activation ):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "ReLU":
        
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
           
    elif activation == "sigmoid":
        
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db
#%%
def backward_propagation(AL, Y, caches,activation_list, lambd= 0, keep_prob = 1, Y_start = 0):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    K, m = AL.shape
    
    # Initializing the backpropagation
    
    if activation_list[-1]== "sigmoid":
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    elif activation_list[-1] == "softmax":
        #dAL = AL
        I = np.zeros((K,m)); I[Y-Y_start,range(m)] = 1
        dAL = I
        
   
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation_list[L-1])
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 

        if keep_prob < 1:
            current_cache = caches[l][:2]
            grads["dA" + str(l + 2)] *= caches[l][2]   # Step 1: Apply mask D  to shut down the same neurons as during the forward propagation
            grads["dA" + str(l + 2)] /= keep_prob           # Step 2: Scale the value of neurons that haven't been shut down
        else:
            current_cache = caches[l]
        
        
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation_list[l])
        
        grads["dA" + str(l + 1)] = dA_prev_temp 
        
        if lambd ==0: 
            grads["dW" + str(l + 1)] = dW_temp
        else:
            grads["dW" + str(l + 1)] = dW_temp + current_cache[0][1]*lambd/m # adding l2 regularization term
            
        grads["db" + str(l + 1)] = db_temp
        
    return grads
#%%
def initialize_optimizer(parameters,optimizer):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
    
    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
                    
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...
    """
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    if optimizer == 'gd':
        pass  # no initialization required for gradient descent
    
    elif optimizer == 'momentum':
        # Initialize velocity
        for l in range(L):
            v["dW" + str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape))
            v["db" + str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape))
            
    elif optimizer == 'adam': 
        # Initialize v, s. Input: "parameters". Outputs: "v, s".
        for l in range(L):
            v["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape))
            v["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape))
            s["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape))
            s["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape))
            
    return v, s
#%%
def update_parameters(optimizer, parameters, grads, learning_rate, v, beta, s, t,
                      beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
        
    v -- python dictionary containing the current velocity (first gradient):
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    v -- python dictionary containing your updated velocities(moving average of the first gradient)
    
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. 
    if optimizer =='gd':
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

    elif optimizer =='momentum':
        for l in range(L):
            # compute velocities
            v["dW" + str(l+1)] = beta*v["dW" + str(l+1)] +(1-beta)*grads["dW" + str(l+1)]
            v["db" + str(l+1)] = beta*v["db" + str(l+1)] +(1-beta)*grads["db" + str(l+1)]
            
            # update parameters
            parameters["W" + str(l+1)] -=learning_rate*v["dW" + str(l+1)]
            parameters["b" + str(l+1)] -=learning_rate*v["db" + str(l+1)]
            
    elif optimizer =='adam': 
        v_corrected = {}        # Initializing first moment estimate, python dictionary
        s_corrected = {}        # Initializing second moment estimate, python dictionary
        
        for l in range(L):
            # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)] +(1-beta1)*grads["dW" + str(l+1)]
            v["db" + str(l+1)] = beta1*v["db" + str(l+1)] +(1-beta1)*grads["db" + str(l+1)]
            
            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
            v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-beta1**t)
            v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-beta1**t)
            
            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)] +(1-beta2)*(grads["dW" + str(l+1)])**2
            s["db" + str(l+1)] = beta2*s["db" + str(l+1)] +(1-beta2)*(grads["db" + str(l+1)])**2
            
            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
            s_corrected["dW" + str(l+1)] =  s["dW" + str(l+1)]/(1-beta2**t)
            s_corrected["db" + str(l+1)] =  s["db" + str(l+1)]/(1-beta2**t)
            
            # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
            parameters["W" + str(l+1)] -= learning_rate*v_corrected["dW" + str(l+1)]/(np.sqrt(s_corrected["dW" + str(l+1)])+epsilon)
            parameters["b" + str(l+1)] -= learning_rate*v_corrected["db" + str(l+1)]/(np.sqrt(s_corrected["db" + str(l+1)])+epsilon)
        
    return parameters, v, s
#%%
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)        # To make your "random" minibatches the same as ours
    m = X.shape[1]              # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        
        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        
        mini_batch_X = shuffled_X[:,mini_batch_size*num_complete_minibatches:m]
        mini_batch_Y = shuffled_Y[:,mini_batch_size*num_complete_minibatches:m]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
#%%
def predict(parameters, X, activation_list):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    # Computes probabilities using forward propagation, and classifies based on max predicted probability.
    AL, _ = forward_propagation(X, parameters, activation_list)
    predictions = np.argmax(AL,axis = 0)
    
    return predictions

def performance_metric(Y,X,parameters,activation_list, chunk_size =[]):
    m = X.shape[1] 
    if not chunk_size:
        chunk_size = m
        
    predicted_class = np.zeros((m)) 
    num_complete_chunks = math.floor(m/chunk_size)
    for k in range(0, num_complete_chunks):
        X_chunk = X[:,k*chunk_size:(k+1)*chunk_size] 
        predicted_class[k*chunk_size:(k+1)*chunk_size] = predict(parameters,X_chunk,activation_list)
    
    if m % chunk_size != 0:
        X_chunk = X[:,chunk_size*num_complete_chunks:m]
        predicted_class[chunk_size*num_complete_chunks:m] = predict(parameters,X_chunk,activation_list)
    #predicted_class = predict(parameters, X, activation_list)
    acc = np.mean(predicted_class == Y)
    return acc
#%%
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()
#%%
def model(X, Y, layer_dims, activation_list, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True, print_every = 1000,
          lambd = 0, keep_prob = 1,Y_start = 0,init_type = 'He',init_parameters = [], scale_f =0.01, seed=0 , 
          X_val=[], Y_val=[]):
    """
    L-layer feed forward neural network model which can be run in different optimizer modes.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    layer_dims = [X.shape[0]]+ layer_dims
    #L = len(layer_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10
    
    # Initialize parameters
    parameters = initialize_parameters(layer_dims,init_type, scale_f, seed,init_parameters)

    # Initialize the optimizer
    v, s = initialize_optimizer(parameters, optimizer)
    
    best_so_far = {}
    best_so_far['parameters']  = parameters
    best_so_far['training loss'] = math.inf
    best_so_far['# optimization updates'] = t
    
    if X_val.any():
        best_so_far['validation loss'] = math.inf
        val_costs = []
    
    # Optimization loop
    for i in range(num_epochs):
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:
            
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            AL, caches = forward_propagation(minibatch_X, parameters, activation_list, keep_prob)
            
            # Compute cost
            cost = compute_cost(AL, minibatch_Y, activation_list[-1] , parameters, lambd =0) 

            # Compute accuracy
            #prediction =  np.argmax(AL,axis = 0)
            
            
            if X_val.any() :
                AL_val, _ = forward_propagation(X_val, parameters, activation_list, keep_prob=1)
                validation_loss = compute_cost(AL_val, Y_val, activation_list[-1] , parameters, lambd=0)
                if validation_loss < best_so_far ['validation loss']:
                    best_so_far ['validation loss']  = validation_loss
                    best_so_far ['parameters'] = parameters
                    best_so_far ['training loss'] = cost
                    best_so_far ['# optimization updates'] = t
            else:
                if cost < best_so_far['training loss']:
                    best_so_far ['training loss'] = cost
                    best_so_far ['# optimization updates'] = t
                    best_so_far ['parameters'] = parameters

                
            
            # Backward propagation 
            grads = backward_propagation(AL, minibatch_Y, caches, activation_list, lambd, keep_prob , Y_start)

            # Update parameters
            t = t + 1 # Adam counter
            parameters, v, s = update_parameters(optimizer, parameters, grads, learning_rate, v, beta, s, t, 
                                                 beta1, beta2, epsilon)
            
                
            
        
        # Print the cost every 1000 epoch
        if print_cost and i % print_every == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
        if print_cost and i % print_every == 0:
            costs.append(cost)
            if X_val.any():
                val_costs.append(validation_loss)
                
    # plot the cost
    plt.plot(costs)
    if X_val.any():
        plt.plot(val_costs,'r')
        plt.plot(best_so_far['# optimization updates']/len(minibatches),best_so_far['validation loss'],'*')
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters, best_so_far
#%%
def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in parameters.keys():
        
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys
#%%
def vector_to_dictionary(theta,parameters):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    param = {}
    count = 0
    for key in parameters.keys():
        param[key] = theta[count:count+np.prod(parameters[key].shape)].reshape(parameters[key].shape)
        count += np.prod(parameters[key].shape)
        
    return param
#%%
def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    count = 0
    #LS = set(list(gradients.keys()))-set(list(gradients.keys())[0::3])
    Lg = len(gradients)
    xx = np.array(range(Lg)); yy =np.zeros(Lg,bool);
    yy[::3]=True; yy=~yy;
    v  = np.zeros(len(xx[yy]),int); u = np.zeros(len(xx[yy]),int);
    v[1::2] = 1; u[::2] =1;
    idx = v-xx[yy]-u;
    
    LS  = [list(gradients.keys())[i] for i in idx]
    
    for key in LS:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta
#%%
def gradient_check(parameters, activation_list, gradients, X, Y,Y_start = 0,lambd = 0, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        thetaplus = np.copy(parameters_values)                                                           # Step 1
        thetaplus[i][0] += epsilon                                                                       # Step 2
        AL, _ = forward_propagation(X,vector_to_dictionary(thetaplus,parameters), activation_list)
        J_plus[i] =  compute_cost(AL, Y, activation_list[-1], vector_to_dictionary(thetaplus,parameters), lambd, Y_start)     # Step 3
        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        thetaminus = np.copy(parameters_values)                                                          # Step 1
        thetaminus[i][0] -=  epsilon                                                                     # Step 2  
        AL, _ = forward_propagation(X, vector_to_dictionary(thetaminus,parameters), activation_list)
        J_minus[i] =  compute_cost(AL, Y, activation_list[-1], vector_to_dictionary(thetaminus,parameters), lambd, Y_start)   # Step 3
        
        # Compute gradapprox[i]
        gradapprox[i] =(J_plus[i]-J_minus[i])/(2*epsilon)
  #  plt.figure()
  #  plt.plot(gradapprox)
  #  plt.plot(grad)
    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grad-gradapprox,ord=2)                              # Step 1'
    denominator = np.linalg.norm(grad,ord=2)+np.linalg.norm(gradapprox,ord=2)      # Step 2'
    difference = numerator/denominator                                             # Step 3'

    if difference > 1e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference

#%% Convolutional layer functions
    
def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant')

    
    return X_pad

#%% Conv single step
def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """
    # Element-wise product between a_slice and W. Add bias.
    s = a_slice_prev*W
    # Sum over all entries of the volume s
    Z = np.sum(s)+float(b)

    return Z
#%%
def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape 
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Compute the dimensions of the CONV output volume using the formula. 
    n_H = int((n_H_prev-f+2*pad)/stride)+1
    n_W = int((n_W_prev-f+2*pad)/stride)+1
    
    # Initialize the output volume Z with zeros. 
    Z = np.zeros((m,n_H,n_W,n_C))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):                               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i,:,:,:]             # Select ith training example's padded activation
        for h in range(n_H):                         # loop over vertical axis of the output volume
            for w in range(n_W):                     # loop over horizontal axis of the output volume
                for c in range(n_C):                 # loop over channels (= #filters) of the output volume
                    
                    # Find the corners of the current "slice" 
                    vert_start = h+h*(stride-1)
                    vert_end = h+h*(stride-1)+f
                    horiz_start = w+w*(stride-1)
                    horiz_end = w+w*(stride-1)+f
                    # Use the corners to define the (3D) slice of a_prev_pad
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])

    # Making sure your output shape is correct
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache
#%%
def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))              
    
    ### START CODE HERE ###
    for i in range(m):                           # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                for c in range (n_C):            # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h+h*(stride-1)
                    vert_end = h+f+h*(stride-1)
                    horiz_start = w+w*(stride-1)
                    horiz_end = w+f+w*(stride-1)
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. 
                    a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    
                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. 
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
                        
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    
    # Making sure your output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache
#%%
def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache
    
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))                           
    dW = np.zeros((f,f, n_C_prev, n_C))
    db = np.zeros((1,1,1,n_C))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev,pad)
    
    for i in range(m):                       # loop over the training examples
        
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i,:,:,:]
        da_prev_pad = dA_prev_pad[i,:,:,:]
        
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h+h*(stride-1)
                    vert_end = h+f+h*(stride-1)
                    horiz_start = w+w*(stride-1)
                    horiz_end = w+f+w*(stride-1)
                    
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
                    
        # Set the ith training example's dA_prev to the unpaded da_prev_pad 
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    # Making sure output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db
#%%
def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    
    ### START CODE HERE ### (≈1 line)
    mask = x == np.max(x)
    ### END CODE HERE ###
    
    return mask
#%%
def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from shape (≈1 line)
    (n_H, n_W) = shape
    
    # Compute the value to distribute on the matrix (≈1 line)
    average = dz/(n_H*n_W)
    # Create a matrix where every entry is the "average" value (≈1 line)
    a = average*np.ones((n_H,n_W))
    ### END CODE HERE ###
    
    return a
#%%
def pool_backward(dA, cache, mode = "max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    
    ### START CODE HERE ###
    
    # Retrieve information from cache (≈1 line)
    (A_prev, hparameters) = cache
    
    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C))
    
    for i in range(m):                       # loop over the training examples
        
        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i,:,:,:]
        
        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels (depth)
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h+h*(stride-1)
                    vert_end = h+f+h*(stride-1)
                    horiz_start = w+w*(stride-1)
                    horiz_end = w+f+w*(stride-1)
                    
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        
                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask*dA[i, h, w, c]
                        
                    elif mode == "average":
                        
                        # Get the value a from dA (≈1 line)
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = (f,f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)
                        
    ### END CODE ###
    
    # Making sure your output shape is correct
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev
#%%