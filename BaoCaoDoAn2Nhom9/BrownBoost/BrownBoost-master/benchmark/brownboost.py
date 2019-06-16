import math
import numpy as np
import math
import sklearn
import random
import copy as cp
from sklearn.tree import DecisionTreeClassifier
from scipy import special
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import  linear_model
from sklearn import datasets

class BrownBoost:

    def __init__(self, weakLearner, c=10, nu=0, stop_criterion=0.01):
        self.weakLearner = weakLearner
        self.c = c
        self.nu = nu
        self.stop_criterion = stop_criterion
        self.alphas = []
        self.H = []

    def fit(self, X, Y):
        self.alphas = []
        self.Hs = []
        s = self.c
        R = np.zeros(X.shape[0])
        k=0
        maxK = 10000
        while s > 0 and k < maxK :
            k += 1
            W = np.exp(-(R + s) ** 2 / self.c)
            h = cp.deepcopy(self.weakLearner)
            h.fit(X, Y, sample_weight=W)
            H = h.predict(X)
            gamma = (W * H * Y).sum()
            alpha, t = self.solveEq(R, H, s, gamma, Y)
            R = R + alpha * H * Y
            s = s - t

            self.alphas.append(alpha)
            self.Hs.append(h)

    def predict(self, X):

        Y = np.zeros(X.shape[0])
        for i in range(0, len(self.Hs)):
            Y += self.alphas[i] * self.Hs[i].predict(X)
        return Y

    def solveEq(self, R, H, s, gamma, Y):
        alpha = min([0.25, gamma])
        t = (alpha ** 2) / 3

        A = R + s
        B = H * Y

        k = 0
        maxK = 1000
        variation = self.stop_criterion + 1

        while k < maxK and variation > self.stop_criterion:
            D = A + (alpha * B - t)
            W = np.exp(-(D ** 2) / self.c)
            w = W.sum()
            u = (W * D * B).sum()
            b = (W * B).sum()
            v = (W * D * (B ** 2)).sum()
            e = (special.erf(D / math.sqrt(self.c)) - special.erf(A / math.sqrt(self.c))).sum()
            sqrtPiC = math.sqrt(math.pi * self.c)
            alpha_1 = alpha + (self.c * w * b + sqrtPiC * u * e) / 2 * (u * w - u * b)
            t_1 = t + (self.c * (b ** 2) + sqrtPiC * v * e) / 2 * (u * w - u * b)
            variation = math.sqrt((alpha - alpha_1) ** 2 + (t - t_1) ** 2)

            alpha = alpha_1
            t = t_1

            k += 1

        return alpha, t












lg_regression = linear_model.LogisticRegression(fit_intercept= True)
iris = datasets.load_iris()
X = iris.data[:,:]
Y = iris.target
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    print('Start')
    dt = DecisionTreeClassifier(max_depth=1)
    BB = BrownBoost(lg_regression)
    BB.fit(X, Y)

    mypre = BB.predict(X)
    for x,y in zip(mypre,Y):
        print("True Answer: ",y," /// Predict : ",x)

