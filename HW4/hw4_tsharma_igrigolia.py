import numpy as np
import copy
NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = [64, 64, 64]     #neurons in hidden layers
NUM_OUTPUT = 10


def generate_one_hot_encoding(y, num_classes):
    one_hot_y = np.zeros((y.shape[0], num_classes))
    one_hot_y[np.arange(y.shape[0]), y] = 1
    return one_hot_y

def train_val_split(X, y, split_ratio):
    num_samples = int(split_ratio * X.shape[1])
    #shuffle X
    X = X[:, np.random.permutation(X.shape[1])]
    # add 1 to X features
    # X = np.vstack([X, np.ones((1, X.shape[1]))])

    return X[:, :num_samples], y[:num_samples, :], X[:, num_samples:], y[num_samples:, :]


def softmax(z):
    ## y_hat = exp(z_i)/ sum(z_i) where z_i = x.T @ w_i + b_i
    y_hat = np.zeros(z.shape)

    for i in range(z.shape[0]):
        y_hat[i] = np.exp(z[i])/np.sum(np.exp(z[i]))

    return y_hat


# def grad (X, y, y_hat, w, alpha):    
#     regularized_weights = np.vstack([w[:-1,:], np.zeros((1,w.shape[1]))])   # removing reguarization of bias
#     def_y = 1/X.shape[1] * ( X @ (y_hat - y) )
                
#     return def_y    #[:-1], w[-1]


def cross_entropy_loss(y, y_hat):
    ## cross entropy loss = -1/n sum[i=1:n] (sum[j=1:k] (y_ij * log(y_hat_ij))) + alpha/2 * sum[i=1:k](w_i.T @ w_i)  
    network_loss =  -1/y.shape[0] * np.sum(y * np.log(y_hat + 0.000001))
    return network_loss

    # regularized_loss = alpha/2 * np.sum(W**2)
    # return network_loss+regularized_loss


def accuracy(y_hat, y):
    y_hat_class = y_hat.argmax(axis=1).reshape(-1,1)
    y_class = y.argmax(axis=1).reshape(-1,1)
    return np.sum(y_class==y_hat_class)/y_hat.shape[0]


def evaluate(X, W, b, y):
    y_hat, _, _ = feedforward(NUM_HIDDEN_LAYERS, X, W, b)
    loss = cross_entropy_loss(y, y_hat)
    acc = accuracy(y_hat, y)
    return loss, acc

def relu(Zi):
    return np.where(Zi > 0, Zi, 0)


def feedforward(num_hidden_layers, X, W, b):
    '''
    num_hidden_layers = total num of hidden layers
    X = input 
    y = , W, b
    '''
    z = []
    h = []
    ## perform a forward pass on the network and compute y_hat
    z0 = (W[0].T @ X).T  + np.repeat(b[0][np.newaxis, :], X.T.shape[0], axis=0)    #num_neuron, num_eg
    z.append(z0)
    h0 = relu(z0)
    h.append(h0)

    ## TRAVERSE HIIDEN LAYERES
    for i in range(1, num_hidden_layers):
        zi = h[i-1] @ W[i]  + np.repeat(b[i][np.newaxis, :], h[i-1].shape[0], axis=0)   # (num_features, num_inputs).T X (num_features, Num_classes)
        z.append(zi)
        hi = relu(zi)
        h.append(hi)
    
    ## output (y_hat)
    y_hat = h[num_hidden_layers-1] @ W[num_hidden_layers]  + np.repeat(b[num_hidden_layers][np.newaxis, :], h[num_hidden_layers-1].shape[0], axis=0)
    
    return y_hat, z, h

def relu_prime(Zi):
    return np.where(Zi > 0, 1, 0)

def backprop(X, y, y_hat, Ws, bs, lr, h, z):
    #create copy of old weights and update the new ones based on the old ones
    Ws_new = copy.deepcopy(Ws)
    bs_new = copy.deepcopy(bs)

    g = y_hat-y # num_eg,num_classes

    for i in range(NUM_HIDDEN_LAYERS,0,-1):
        del_b = np.mean(g, axis=0)
        del_W = (h[i-1].T @ g)/y.shape[0]

        #update wts and b for last hidden layer
        Ws_new[i] -= lr * del_W
        bs_new[i] -= lr * del_b
        
        # get relu_prime
        del_relu = relu_prime(z[i-1])

        g = (g @ Ws[i].T) * del_relu
    
    #update the first wts and bs layer
    del_b0 = np.mean(g, axis=0)
    del_W0 = (X @ g)/y.shape[0]  

    Ws_new[0] -= lr * del_W0
    bs_new[0] -= lr * del_b0

    return Ws_new, bs_new




def train(X, y, config):
    # initialize weights + bias vec to random
    Ws, bs = initWeightsAndBiases()
    for epoch in range(config['num_epoch']):
        for idx in range(0, y.shape[0], config['batch_size']):
            #define X_train, y_train
            X_, y_ = X[:, idx:idx+ config['batch_size']], y[idx:idx+config['batch_size'], :]
            
            ## feedforward network
            y_hat, z, h = feedforward(NUM_HIDDEN_LAYERS, X_, Ws, bs)

            ## perform backprop
            Ws, bs = backprop(X_, y_, y_hat, Ws, bs, config['lr'], h, z)


            tr_loss, tr_acc = evaluate(X_, Ws, bs, y_)
            print('Train Loss for config: lr: '+ str(config['lr']) + ' batch:' +str(config['batch_size']) + ' num_epoch: ' + str(config['num_epoch']) + '= ' + str(tr_loss))
                    
    return Ws, bs

## in training
    # perform feedforward
    # compute gradients for backprop


def initWeightsAndBiases ():
    Ws = []
    bs = []

    # Strategy:
    # Sample each weight from a 0-mean Gaussian with std.dev. of 1/sqrt(numInputs).
    # Initialize biases to small positive number (0.01).

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_INPUT, NUM_HIDDEN[0]))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b = 0.01 * np.ones(NUM_HIDDEN[0])
    Ws.append(W)
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN[i], NUM_HIDDEN[i+1]))/NUM_HIDDEN[i]**0.5) - 1./NUM_HIDDEN[i]**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN[i+1])
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_HIDDEN[-1], NUM_OUTPUT))/NUM_HIDDEN[-1]**0.5) - 1./NUM_HIDDEN[-1]**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return Ws, bs



def train_age_regressor():
    # Load data
    X_tr = np.reshape(np.load("HW3/fashion_mnist_train_images.npy"), (-1, 28*28)).T
    ytr = np.load("HW3/fashion_mnist_train_labels.npy")
    X_te = np.reshape(np.load("HW3/fashion_mnist_test_images.npy"), (-1, 28*28)).T
    yte = np.load("HW3/fashion_mnist_test_labels.npy")

    # normalize X else gadients would explode
    X_tr /= 255
    X_te /= 255
    

    # generate one hot encoding
    ytr = generate_one_hot_encoding(ytr,10)
    yte = generate_one_hot_encoding(yte,10)

    # print(X_tr.shape,ytr.shape,X_te.shape,yte.shape)

    # split x train, y train
    X_train, y_train, X_val, y_val = train_val_split(X_tr, ytr, 0.8)
    # define hyper param list for num_iters, lr, batch_size, regularization term
    hyperparams = {
        'num_epoch_list': [10],
        'lr_list': [ 0.1],
        'batch_size_list': [128],
        # 'reg_list': [0.000001, 0.00001, 0.0001],
    }

    optimal_loss = np.inf
    optimal_config = {}

    for lr in hyperparams["lr_list"]:
        for batch_size in hyperparams['batch_size_list']:
            # for alpha in hyperparams['reg_list']:
                for num_epoch in hyperparams['num_epoch_list']:
                    #train
                    Ws, bs = train(X_train, y_train, {'lr':lr, 'batch_size':batch_size, 'num_epoch':num_epoch})
                    
                    ## after training perform hyperparam tuning
                    current_config_loss, curr_acc = evaluate(X_val, Ws, bs, y_val)

                    print('Val Loss for config: lr: '+ str(lr) + ' batch:' +str(batch_size) +  ' num_epoch: ' + str(num_epoch) + '= ' + str(current_config_loss))
                    print('Val Acc : '+  str(curr_acc))
                    
                    if current_config_loss<optimal_loss:
                        optimal_loss = current_config_loss

                        optimal_config["lr"]= lr
                        optimal_config['batch_size']= batch_size
                        # optimal_config['alpha']= alpha
                        optimal_config['num_epoch']= num_epoch


    weights_optimal, bias_optimal = train(X_tr, ytr, optimal_config)
    print('Optimal config: ' + str(optimal_config))
    

    # w, b = linear_regression(X_tr, ytr)
    # Report fMSE cost on the training and testing data (separately)
    loss_tr, acc_tr = evaluate(X_tr, weights_optimal, bias_optimal, ytr)
    loss_te, acc_te = evaluate(X_te, weights_optimal, bias_optimal, yte)
    
    print("Unregularized MSE Loss on training set: ", loss_tr )
    print("Unregularized MSE Loss on test set: ", loss_te)
    print("Unregularized Accuracy on training set: ", acc_tr )
    print("Unregularized Accuracy on test set: ", acc_te)


train_age_regressor()