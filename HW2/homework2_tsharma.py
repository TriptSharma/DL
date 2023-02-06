import numpy as np

#########################################################################################
################################### QUESTION 1 ##########################################
#########################################################################################


def doCrossValidation (D, k, H):
    allIdxs = np.arange(len(D))
    # Randomly split dataset into k folds
    idxs = np.random.permutation(allIdxs)
    idxs = idxs.reshape(k, -1)
    # define optimal_accuracy, optimal_h
    optimal_acc = 0
    optimal_h = H[0]
    for h in H:
        accuracies = []
        for fold in range(k):
            # Get all indexes for this fold
            testIdxs = idxs[fold,:]
            # Get all the other indexes
            trainIdxs = np.array(set(allIdxs) - set(testIdxs)).flatten()
            # Train the model on the training data
            model = trainModel(D[trainIdxs], h)
            # Test the model on the testing data
            accuracies.append(testModel(model, D[testIdxs]))
        
        curr_accuracy = np.mean(accuracies)
        if(curr_accuracy>optimal_acc):
            optimal_h = h

    return optimal_h

def doDoubleCrossValidation (D, k, H):
    allIdxs = np.arange(len(D))
    idxs = np.random.permutation(allIdxs)
    idxs = idxs.reshape(k**2, -1)
    accuracies = []
    for outerfold in range(0,k**2,k):
        testIdxs = idxs[outerfold:k+outerfold,:] 
        trainIdxs = np.array(set(allIdxs) - set(testIdxs)).flatten()
        
        # get optimal h
        h = doCrossValidation(trainIdxs, k, H)
        # outerTrainIdxs = np.array(set(allTrainIdxs) - set(innerTrainIdxs)).flatten()
        model = trainModel(D[trainIdxs], h)
        accuracies.append(testModel(model, D[testIdxs]))
        return np.mean(accuracies)


#########################################################################################
################################### QUESTION 3 ##########################################
#########################################################################################

# add 1 to X_tr
        # CLOSED FORM SOLUTION
        # (X . X.T)^-1 . w =  (X . y) where X is of shape (num of features, num of inputs)
        # w = np.linalg.solve(X.T @ X, (X.T @ y_tr))

################# GRADIENT DESCENT ################

# repeat till #max_iters or loss<thresh_loss
    # optimize w 
    # compute grad = 1/n * X.T ( X . w - y )
    #! OR.....


def train_val_split(X, y, split_ratio):
    num_samples = int(split_ratio * X.shape[1])
    #shuffle X
    X = X[:, np.random.permutation(X.shape[1])]
    # add 1 to X features
    X = np.vstack([X, np.ones((1, X.shape[1]))])

    return X[:, :num_samples], y[:num_samples].reshape((-1,1)), X[:, num_samples:], y[num_samples:].reshape((-1,1))


def grad (X, y, w, alpha):    
    regularized_weights = np.append(w[:-1], 0)[:,np.newaxis]
    gradient = 1/X.shape[1] * (X @ ((X.T @ w) - y) + alpha * regularized_weights)
        # NOTE: w contains both, weights and bias
    return gradient    #[:-1], w[-1]


def evaluate(X, w, y):
    loss = np.mean(( X.T @ w - y)**2)/2
    return loss

def train(X, y, config):
    # initialize weights + bias vec to random
    w = np.random.rand(X.shape[0],1)
    for epoch in range(config['num_epoch']):
        for idx in range(0, y.shape[0], config['batch_size']):
            #define X_train, y_train
            X_, y_ = X[:, idx:idx+ config['batch_size']], y[idx:idx+config['batch_size']]
            
            # compute gradient of loss wrt w
            del_loss_wrt_w = grad(X_, y_, w, config['alpha'])

            # update weights
            w = w - config['lr'] * del_loss_wrt_w
            # w_ = w - lr * grad
    return w


def train_age_regressor():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48)).T
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48)).T
    yte = np.load("age_regression_yte.npy")

    # print(X_tr.shape,ytr.shape,X_te.shape,yte.shape)

    # split x train, y train
    X_train, y_train, X_val, y_val = train_val_split(X_tr, ytr, 0.8)
    # define hyper param list for num_iters, lr, batch_size, regularization term
    hyperparams = {
        'num_epoch_list': [25,50,100],
        'lr_list': [0.0005, 0.001, 0.003],
        'batch_size_list': [16,32,64],
        'reg_list': [0.001, 0.01, 0.1],
    }

    optimal_loss = np.inf
    optimal_config = {}

    for lr in hyperparams["lr_list"]:
        for batch_size in hyperparams['batch_size_list']:
            for alpha in hyperparams['reg_list']:
                for num_epoch in hyperparams['num_epoch_list']:
                    #train
                    w = train(X_train, y_train, {'lr':lr, 'batch_size':batch_size, 'alpha':alpha,'num_epoch':num_epoch})
                                    
                    ## after training perform hyperparam tuning
                    current_config_loss = evaluate(X_val, w, y_val)

                    print('Val Loss for config: lr: '+ str(lr) + ' batch:' +str(batch_size) + ' alpha:' + str(alpha) + ' num_epoch: ' + str(num_epoch) + '= ' + str(current_config_loss))
                    if current_config_loss<optimal_loss:
                        optimal_loss = current_config_loss

                        optimal_config["lr"]= lr
                        optimal_config['batch_size']= batch_size
                        optimal_config['alpha']= alpha
                        optimal_config['num_epoch']= num_epoch


    w_optimal = train(np.vstack([X_tr, np.ones((1, X_tr.shape[1]))]), ytr.reshape(-1,1), optimal_config)
    weights_optimal, bias_optimal = w_optimal[:-1], w_optimal[-1]
    print('Optimal config: ' + str(optimal_config))
    

    # w, b = linear_regression(X_tr, ytr)
    # Report fMSE cost on the training and testing data (separately)
    loss_tr = np.mean(( X_tr.T @ weights_optimal + bias_optimal - ytr)**2)/2
    loss_te = np.mean(( X_te.T @ weights_optimal + bias_optimal - yte)**2)/2

    print("Unregularized MSE Loss on training set: ", loss_tr )
    print("Unregularized MSE Loss on test set: ", loss_te)


train_age_regressor()
