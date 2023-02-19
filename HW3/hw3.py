import numpy as np

def generate_one_hot_encoding(y, num_classes):
    one_hot_y = np.zeros((y.shape[0], num_classes))
    one_hot_y[np.arange(y.shape[0]), y] = 1
    return one_hot_y

def train_val_split(X, y, split_ratio):
    num_samples = int(split_ratio * X.shape[1])
    #shuffle X
    X = X[:, np.random.permutation(X.shape[1])]
    # add 1 to X features
    X = np.vstack([X, np.ones((1, X.shape[1]))])

    return X[:, :num_samples], y[:num_samples, :], X[:, num_samples:], y[num_samples:, :]


def softmax(X, W, y):
    ## y_hat = exp(z_i)/ sum(z_i) where z_i = x.T @ w_i + b_i
    z = X.T @ W     # (num_features, num_inputs).T X (num_features, Num_classes)
    y_hat = np.zeros(z.shape)

    for i in range(y.shape[0]):
        y_hat[i] = np.exp(z[i])/np.sum(np.exp(z[i]))

    return y_hat
    

def grad (X, y, y_hat, w, alpha):    
    regularized_weights = np.vstack([w[:-1,:], np.zeros((1,w.shape[1]))])
    gradient = 1/X.shape[1] * ( X @ (y_hat - y) + alpha * regularized_weights)
        # NOTE: w contains both, weights and bias
    return gradient    #[:-1], w[-1]


def cross_entropy_loss(y, y_hat, alpha, W, isRgularized=True):
    ## cross entropy loss = -1/n sum[i=1:n] (sum[j=1:k] (y_ij * log(y_hat_ij))) + alpha/2 * sum[i=1:k](w_i.T @ w_i)  
    network_loss =  -1/y.shape[0] * np.sum(y * np.log(y_hat + 0.000001))
    if not isRgularized:
        return network_loss

    regularized_loss = alpha/2 * np.sum(W**2)
    return network_loss+regularized_loss


def accuracy(y_hat, y):
    y_hat_class = y_hat.argmax(axis=1).reshape(-1,1)
    y_class = y.argmax(axis=1).reshape(-1,1)
    return np.sum(y_class==y_hat_class)/y_hat.shape[0]


def evaluate(X, W, y, alpha):
    y_hat = softmax(X, W, y)
    loss = cross_entropy_loss(y, y_hat, alpha, W)
    acc = accuracy(y_hat, y)
    return loss, acc

def train(X, y, config):
    # initialize weights + bias vec to random
    W = np.random.rand(X.shape[0],y.shape[1])
    for epoch in range(config['num_epoch']):
        for idx in range(0, y.shape[0], config['batch_size']):
            #define X_train, y_train
            X_, y_ = X[:, idx:idx+ config['batch_size']], y[idx:idx+config['batch_size'], :]
            
            # compute gradient of loss wrt w
            y_hat = softmax(X_, W, y_)
            del_loss_wrt_w = grad(X_, y_, y_hat, W, config['alpha'])

            # update weights
            W = W - config['lr'] * del_loss_wrt_w
            # w_ = w - lr * grad
    return W


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
        'num_epoch_list': [25, 50, 100],
        'lr_list': [0.05, 0.03, 0.1],
        'batch_size_list': [500,1000,2000],
        'reg_list': [0.000001, 0.00001, 0.0001],
    }

    optimal_loss = np.inf
    optimal_config = {}

    for lr in hyperparams["lr_list"]:
        for batch_size in hyperparams['batch_size_list']:
            for alpha in hyperparams['reg_list']:
                for num_epoch in hyperparams['num_epoch_list']:
                    #train
                    W = train(X_train, y_train, {'lr':lr, 'batch_size':batch_size, 'alpha':alpha,'num_epoch':num_epoch})
                                    
                    ## after training perform hyperparam tuning
                    current_config_loss, curr_acc = evaluate(X_val, W, y_val, alpha)

                    print('Val Loss for config: lr: '+ str(lr) + ' batch:' +str(batch_size) + ' alpha:' + str(alpha) + ' num_epoch: ' + str(num_epoch) + '= ' + str(current_config_loss))
                    print('Val Acc : '+  str(curr_acc))
                    
                    if current_config_loss<optimal_loss:
                        optimal_loss = current_config_loss

                        optimal_config["lr"]= lr
                        optimal_config['batch_size']= batch_size
                        optimal_config['alpha']= alpha
                        optimal_config['num_epoch']= num_epoch


    w_optimal = train(np.vstack([X_tr, np.ones((1, X_tr.shape[1]))]), ytr, optimal_config)
    weights_optimal, bias_optimal = w_optimal[:-1], w_optimal[-1]
    print('Optimal config: ' + str(optimal_config))
    

    # w, b = linear_regression(X_tr, ytr)
    # Report fMSE cost on the training and testing data (separately)
    loss_tr, acc_tr = evaluate(np.vstack([X_tr, np.ones((1, X_tr.shape[1]))]), w_optimal, ytr, optimal_config['alpha'])
    loss_te, acc_te = evaluate(np.vstack([X_te, np.ones((1, X_te.shape[1]))]), w_optimal, yte, optimal_config['alpha'])
    
    print("Unregularized MSE Loss on training set: ", loss_tr )
    print("Unregularized MSE Loss on test set: ", loss_te)
    print("Unregularized Accuracy on training set: ", acc_tr )
    print("Unregularized Accuracy on test set: ", acc_te)


train_age_regressor()