"""
The following sources were used during the study.
1) https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2
2) https://developer.ibm.com/articles/implementing-logistic-regression-from-scratch-in-python/
"""

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, lr=0.01,n_iters=100):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None




    @staticmethod
    def train_test_split(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))


    def log_fit(self, X, y):
        num_sample, num_feat = X.shape
        self.weights = np.zeros(num_feat)
        self.bias = 0

        for _ in range(self.n_iters):
            z = np.dot(X, self.weights) + self.bias 
            pred = LogisticRegression.sigmoid(z)

            gradient_weight = (1/num_sample)*np.dot(X.T, (pred-y))
            gradient_bias = (1/num_sample)*np.sum(pred-y)

            self.weights -= self.lr*gradient_weight
            self.bias -= self.lr*gradient_bias

            

    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias 
        y_pred = LogisticRegression.sigmoid(z)

        return np.round(y_pred,2)



    def accuracy(self,y_test, y_pred):
        accuracy = (y_pred == y_test).mean()
        return accuracy

            



