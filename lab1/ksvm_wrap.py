import numpy as np
import data
from sklearn.svm import SVC
import matplotlib.pyplot as plt

class KSVMWrap:

    def __init__(self,X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.model = SVC(C=param_svm_c, gamma=param_svm_gamma, probability=True)
        self.model.fit(X, Y_)

    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_scores(self, X):
        return self.model.predict_log_proba(X)

    def support(self):
        return self.model.support_

if __name__ == "__main__":
    np.random.seed(100)

    X,Y_ = data.sample_gmm_2d(6, 2, 10)

    model = KSVMWrap( X, Y_)

    # dohvati vjerojatnosti na skupu za učenje
    probs = model.get_scores(X)
    Y = np.argmax(probs, axis=1)
    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, precision, confusion_matrix = data.eval_perf_multi(Y, Y_)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Confusion matrix:\n', confusion_matrix)


    # iscrtaj rezultate, decizijsku plohu
    rect=(np.min(X, axis=0), np.max(X, axis=0))
    decision = lambda X: model.get_scores(X)[:,0]
    data.graph_surface(decision, rect, offset=0)
    
    # graph the data points
    data.graph_data(X, Y_, Y, special=model.support())
    plt.show()
