import numpy as np
def initialize_parameters(n_a,n_x,n_y) : 
    np.random.seed(1) 
    Wax = np.random.randn(n_a,n_x)*0.01
    Waa = np.random.randn(n_a,n_a)*0.01 
    Wya = np.random.randn(n_y,n_a) *0.01
    b = np.zeros((n_a,1))
    by = np.zeros((n_y,1)) 
    parameters = {"Wax":Wax,"Waa":Waa,"Wya":Wya,"b":b,"by":by}
    return parameters 
def initialize_loss(vocab_size,seq_length) : 
    return -np.log(1/vocab_size) * seq_length 
def softmax(x) : 
    ex = np.exp(x-np.max(x)) 
    return ex/ex.sum(axis=0) 
def rnn_step_forward(parameters,a_prev,x) : 
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b'] 
    a_next = np.tanh(np.dot(Wax,x) + np.dot(Waa,a_prev) + b) 
    p_t = softmax(np.dot(Wya,a_next) + by)
    return a_next,pt
def rnn_forward(X, Y, a0, parameters, vocab_size = 27) : 
    x,a,y_hat = {},{},{} 
    a[-1] = np.copy(a0) # like a0, random
    loss = 0 
    for t in range(len(X)) : 
        x[t] = np.zeros((vocab_size,1)) 
        if (X[t] != None ) : 
           x[t][X[t]] = 1 
        a[t],y_hat[t] = rnn_step_forward(parameters,a[t-1],x[t])
        loss -= np.log(y_hat[t][Y[t],0]) 
    cache = (y_hat,a,x)
    return loss,cache
def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] # backprop into h
    daraw = (1 - a * a) * da # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients
def rnn_backward(X,Y,paramters,cache) : 
    gradients = {} 
    (y_hat,a,x) = cache 
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    gradients['dWax'],gradients['dWaa'],gradients]'dWya'] = np.zeros_like(Wax),np.zeros_like(Waa),np.zeros_like(Wya) 
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by) 
    gradients['da_next'] = np.zeros_like(a[0]) 
    for t in reversed(range(len(X))) : 
        dy = np.copy(y_hat[t]) 
        dy[Y[t]] -=1 
        gradients = rnn_step_backward(dy,gradients,parameters,x[t],a[t],a[t-1])
    return gradient,a
def update_parameters(parameters, gradients, lr):

    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b']  += -lr * gradients['db']
    parameters['by']  += -lr * gradients['dby']
    return parameters