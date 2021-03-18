import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        self.X = X
        self.y = y

        return

    def minkDistance(self, a, b):
        result = np.sum(((np.absolute(a - b)) ** self.p)) ** 1 / self.p

        return result

    def euclDistance(self, a, b):
        result = np.sqrt(np.sum((a - b) ** 2))

        return result

    def manhDistance(self, a, b):
        result = np.sum(np.absolute(a - b))

        return result

    def cosiDistance(self, a, b):
        dotProduct = np.dot(a, b)
        dotX = (np.dot(a, a) ** .5)
        dotx = (np.dot(b, b) ** .5)
        result = 1 - dotProduct / (dotX * dotx)

        return result

    def dist(self, x):
        # Calculate distances of training data to a single input data point (distances from self.X to x)
        # Output np.array([distances to x])
        if self.metric == "minkowski":
            AllOfX = self.X[self.X.columns]
            distToX = [self.minkDistance(x, b) for b in AllOfX.to_numpy()]

        elif self.metric == "euclidean":
            AllOfX = self.X[self.X.columns]
            distToX = [self.euclDistance(x, b) for b in AllOfX.to_numpy()]

        elif self.metric == "manhattan":
            AllOfX = self.X[self.X.columns]
            distToX = [self.manhDistance(x, b) for b in AllOfX.to_numpy()]

        elif self.metric == "cosine":
            AllOfX = self.X[self.X.columns]
            distToX = [self.cosiDistance(x, b) for b in AllOfX.to_numpy()]

        else:
            raise Exception("Unknown criterion.")

        return distToX

    def k_neighbors(self,x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors) e.g. {"Class A":3, "Class B":2}
        distToX = self.dist(x)
        kValues = np.argsort(distToX)[:self.n_neighbors]
        kLabel = [self.y[k] for k in kValues]
        output = Counter(kLabel)

        return output

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]

        return predictions


    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        probs = []
        try:
            X_feature = X[self.X.columns]
        except:
            raise Exception("Input data mismatch.")

        for x in X_feature.to_numpy():
            neighbors = self.k_neighbors(x)
            # Calculate the probability of data point x belonging to each class
            # e.g. prob = {"2": 1/3, "1": 2/3}
            prob = {k: neighbors[k] / float(self.n_neighbors) for k in self.classes_}
            probs.append(prob)
        probs = pd.DataFrame(probs, columns=self.classes_)

        return probs