# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 08:54:16 2017

@author: Eyas
"""
import numpy as np
#%%
def initialize_parameters(layer_dims, init_type = 'random' , scale_f = 0.01, seed = 0):
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
        
    for l in range(1, L):
        
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * sF(layers_dims[l-1])
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
    #s = np.exp(z)/sum(np.exp(z));
    cache = z
    
    return s , cache

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
        activation_cache.append(D)
        
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
    
    
    assert(AL.shape == (1,X.shape[1]))
            
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
    if cost_type == "bernoulli":
        cost = -float(np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))/m)
        #cost = -np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))/m
    elif cost_type =="multinoulli" and onehot==False:
        #cost = -np.sum(np.log(AL).ravel()[(Y-Y_start)*K+range(m)])/m
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

    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis=1,keepdims= True)/m
    dA_prev = np.dot(W.T,dZ)

    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db
#%%
def relu_backward(dA, activation_cache):
    
    dZ = dA*(activation_cache>0)
    
    return dZ
#%%
def sigmoid_backward(dA, activation_cache):
    
    dZ = dA*activation_cache*(1- activation_cache)
    
    return dZ

#%%
def softmax_backward(dAL,activation_cache):    

    dZ = activation_cache - dAL
    
    return dZ

#%%
def log_sum_exp(Z):
    
    max_z = np.max(Z,0)
    
    log_s_e = np.log(np.sum(np.exp(Z -max_z),0)) + max_z
    
    return log_s_e
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
def backward_propagation(AL, Y, caches, cost_type ,activation_list, lambd= 0, keep_prob = 1, Y_start = 0):
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
    
    if cost_type == "bernoulli":
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    elif cost_type == "multinoulli":
        I = np.zeros(K,m); I[Y-Y_start,range(m)] = 1
        dAL = I
    
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation_list[L-1])
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
    
        if keep_prob < 1:
            grads["dA" + str(l + 2)] *= current_cache[1][-1] # Step 1: Apply mask D  to shut down the same neurons as during the forward propagation
            grads["dA" + str(l + 2)] /= keep_prob            # Step 2: Scale the value of neurons that haven't been shut down
        
        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation_list[l + 1])
        
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
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
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
    shuffled_Y = Y[:, permutation].reshape((1,m))

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
    AL, cache = forward_propagation(X, parameters, activation_list)
    predictions = np.argmax(AL,0)
    
    return predictions
#%%
def model(X, Y, layers_dims, activation_list , optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True,
          lambd = 0, keep_prob = 1):
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

    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10
    
    # Initialize parameters
    parameters = initialize_parameters(layers_dims,init_type = 'He', scale_f =0.01, seed=0)

    # Initialize the optimizer
    v, s = initialize_optimizer(parameters, optimizer)
    
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
            cost = compute_cost(AL, minibatch_Y, cost_type, parameters, lambd) 

            # Backward propagation 
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # Update parameters
            t = t + 1 # Adam counter
            parameters, v, s = update_parameters(optimizer, parameters, grads, learning_rate, v, beta, s, t, 
                                                 beta1, beta2, epsilon)
        
        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
                
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters

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
def gradient_check(parameters, activation_list,cost_type, gradients, X, Y,Y_start = 0,lambd = 0, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
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
        AL, _ = forward_propagation(X,vector_to_dictionary(thetaplus, parameters), activation_list)
        J_plus[i] =  compute_cost(AL, Y, cost_type, vector_to_dictionary(thetaplus, parameters), lambd, Y_start)     # Step 3
        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        thetaminus = np.copy(parameters_values)                                                          # Step 1
        thetaminus[i][0] -=  epsilon                                                                     # Step 2  
        AL, _ = forward_propagation(X, vector_to_dictionary(thetaminus, parameters), activation_list)
        J_minus[i] =  compute_cost(AL, Y, cost_type, vector_to_dictionary(thetaminus, parameters), lambd, Y_start)   # Step 3
        
        # Compute gradapprox[i]
        gradapprox[i] =(J_plus[i]-J_minus[i])/(2*epsilon)
        
    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grad-gradapprox,ord=2)                              # Step 1'
    denominator = np.linalg.norm(grad,ord=2)+np.linalg.norm(gradapprox,ord=2)      # Step 2'
    difference = numerator/denominator                                             # Step 3'

    if difference > 1e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference