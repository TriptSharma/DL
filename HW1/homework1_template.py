import numpy as np

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    # AB - C
    return (A @ B) - C

def problem_1c (A, B, C):
    # (Hadamard  prd) elementwise prd of A, B + C.T
    return A*B + C.T

def problem_1d (x, y):
    # get inner product (results in scalar)
    return x.T @ y

def problem_1e (A, b):
    # np.solve solves x for linear set of equations:    Ax = b 
    # i.e.      x = A^-1 b
    # where A = square matrix with full rank
    x = np.linalg.solve(A,b)
    return x

def problem_1f (A, i):
    # return the sum of all the entries in the ith row whose column index is even
    mask = [A[i,j]%2 for j in range(A[i].shape[0])]
    return np.sum(A[i,:], where=mask)

def problem_1g (A, c, d):
    # given matrix A and scalars c, d, 
    # compute the arithmetic mean over all entries of A that are between c and d (inclusive)
    mask = np.where(A>=c and A<=d, A, 0)
    mean = np.mean(mask[np.nonzero(mask)])  #get the non zero values from the mask and compute there mean
    return mean

def problem_1h (A, k):
    ## TODO: check which are top K eigenvecs
    # return top k right-eigenvectors of A 
    w, X = np.linalg.eig(A) #w = eigenval, X = eiegnvec
    return X[:,:k]

def problem_1i (x, k, m, s):
    n = x.shape[0]
    #create multi-dim gaussian
        # create the mean vector
    mu = x + m*np.ones((n,))
        # create the std dev matrix
    sigma = np.eye(n) * s
    # get n*k random values from the gauss distrin
    # reshape to (n,k)
    result = np.random.multivariate_normal(mu, sigma, (n,k))
    return result

def problem_1j (A):
    #return a matrix that results from randomly permuting the columns
    cols = A.shape[1]
    permutation = np.random.randint(0,cols,cols)
    return A[:, permutation]

def problem_1k (x):
    # return (xi − x)/σ
    return (x - np.mean(x))/np.std(x)

def problem_1l (x, k):
    # return a n × k matrix consisting of k copies of x
    return np.repeat(x, k, axis=1)

def problem_1m (X, Y):
    # return l2 distance between col vectors of 2 matrices of size (k,n) and (k,m)
    vec_diff = X-Y[:,None]
    dist = np.sqrt(np.sum(vec_diff**2, axis=-1))
    return dist.T

def problem_1n (matrices):
    count = 0
    sizes = [mat.shape for mat in matrices]
    curr_size = sizes[0]
    for i in range(1,len(sizes)):
        count += curr_size[0]*curr_size[1]*sizes[i][1]
        curr_size = [curr_size[0],sizes[i][1]]
    return curr_size

def linear_regression (X_tr, y_tr):
    # add 1 to X_tr
    X = np.hstack([X_tr, np.ones((X_tr.shape[0],1))])
    
    # (X . X.T)^-1 . w =  (X . y) where X is of shape (num of features, num of inputs)
    w = np.linalg.solve(X.T @ X, (X.T @ y_tr))

    #! OR.....
    # initialize weights + bias vec to random
    # optimize w 
    # compute grad = 1/n * X.T ( X . w - y )
        # w_ = w - lr * grad
    # repeat till #max_iters or loss<thresh_loss
    return w[:-1], w[-1]

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("HW1/age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("HW1/age_regression_ytr.npy")
    X_te = np.reshape(np.load("HW1/age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("HW1/age_regression_yte.npy")

    # print(X_tr.shape,ytr.shape,X_te.shape,yte.shape)

    w, b = linear_regression(X_tr, ytr)

    # Report fMSE cost on the training and testing data (separately)
    loss_tr = np.mean(( X_tr @ w + b - ytr)**2)/2
    loss_te = np.mean(( X_te @ w + b - yte)**2)/2

    print("MSE Loss on training set: ", loss_tr )
    print("MSE Loss on test set: ", loss_te)

A = np.array([[1,2,3],[2,3,4],[4,5,6]])
B = np.array([[1,2,3],[2,3,4],[4,5,6]])
C = np.array([[1,2,3],[2,3,4],[4,5,6]])

train_age_regressor()



print(problem_1b)
print(A)
print(B)
print(C)
