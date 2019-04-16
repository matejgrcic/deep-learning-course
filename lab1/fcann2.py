import numpy as np
import data
import matplotlib.pyplot as plt

def iverson_fn(x):
    return 1 if x > 0 else 0
iverson = np.vectorize(iverson_fn)

def fcann2_train(X, Y_, param_niter = 100000, param_delta = 0.05, param_lambda = 1e-3, H = 5):
    D = X.shape[1]
    N = X.shape[0]
    C = int(np.max(Y_) + 1)
    
    W1 = 0.01 * np.random.randn(D,H)
    b1 = np.zeros((1,H))
    W2 = 0.01 * np.random.randn(H,C)
    b2 = np.zeros((1,C))

    for i in range(param_niter):
        h1 = np.maximum(0, np.matmul(X, W1) + b1) 
        scores = np.matmul(h1, W2) + b2
        
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x C]
        
        data_loss = np.sum(-np.log(probs[range(N), Y_])) / N
        reg_loss = param_lambda * 0.5 * (np.linalg.norm(W1) + np.linalg.norm(W2))
        loss = data_loss + reg_loss
        if i % 1000 == 0:
            print(f"iteration {i}: loss {loss}")
        
        dscores = probs
        dscores[range(N), Y_] -= 1
        dscores /= N
        
        dW2 = np.dot(h1.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        dh1 = np.dot(dscores, W2.T)
        dh1[h1 <= 0] = 0
        # finally into W,b
        dW1 = np.dot(X.T, dh1)
        db1 = np.sum(dh1, axis=0, keepdims=True)
        dW2 += param_lambda * W2
        dW1 += param_lambda * W1
        
        b2 += -param_delta * db2
        W2 += -param_delta * dW2
        W1 += -param_delta * dW1
        b1 += -param_delta * db1
    
    return W1, W2, b1, b2

def logreg_classify(X, W1, W2, b1, b2):
    N = X.shape[0]
    h1 = np.maximum(0, np.dot(X, W1) + b1)
    scores = np.dot(h1, W2) + b2
    
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


if __name__ == "__main__":
    np.random.seed(100)

    X,Y_ = data.sample_gmm_2d(6, 2, 10)
    W1, W2, b1, b2 = fcann2_train(X, Y_)
    probs = logreg_classify(X, W1, W2, b1, b2)
    Y = np.argmax(probs, axis=1)
    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, precision, confusion_matrix = data.eval_perf_multi(Y, Y_)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Confusion matrix:\n', confusion_matrix)

    # iscrtaj rezultate, decizijsku plohu
    rect=(np.min(X, axis=0), np.max(X, axis=0))
    decision = lambda X: logreg_classify(X, W1, W2, b1, b2)[:,0]
    data.graph_surface(decision, rect, offset=0)
    
    # graph the data points
    data.graph_data(X, Y_, Y, special=[])
    plt.show()

