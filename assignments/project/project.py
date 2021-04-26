import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils import resample
from gensim.parsing.preprocessing import STOPWORDS
sys.path.insert(0, '..')
from assignments.assignment8 import my_evaluation


class my_model():

    def removeStopWords(self, data_frame, column_name):

        data_frame[column_name] = data_frame[column_name].apply(
            lambda variable: " ".join([i for i in variable.lower().split() if i not in self.all_genism_stop_words]))

    def removeSpecialChar(self, data_frame, columns):

        data_frame.columns = data_frame.columns.str.replace('[!,@,#,$,%,^,&,*,\",:,;,.]', '')

    def preprocess(self, data_frame):

        data_frame['location'] = data_frame.location.fillna('None')
        data_frame['description'] = data_frame.description.fillna('Not specified')
        data_frame['requirements'] = data_frame.description.fillna('Not specified')
        data_frame['has_company_logo'] = data_frame.has_company_logo.map({1: 't', 0: 'f'})
        data_frame['title'] = data_frame.title.str.replace(r'<[^>]*>', '')
        data_frame['description'] = data_frame.description.str.replace(r'<[^>]*>', '')
        data_frame['requirements'] = data_frame.requirements.str.replace(r'<[^>]*>', '')
        data_frame.drop(['telecommuting', 'has_questions'], axis=1, inplace=True)

        for values in data_frame.columns:
            data_frame[values] = data_frame[values].str.replace(r'\W',' ').str.replace(r'\s$', '')
        self.all_genism_stop_words = STOPWORDS
        columnsWText = list(data_frame.columns.values)
        for values in columnsWText:
            self.removeStopWords(data_frame, values)
        for values in columnsWText:
            self.removeSpecialChar(data_frame, values)
        return data_frame


    def fit(self, X, y):

        X = self.preprocess(X)
        self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=True, smooth_idf=True, ngram_range=(1, 5))

        XX = self.preprocessor.fit_transform(X["description"], X["requirements"])

        self.sgd = SGDClassifier()
        sgd_grid = {'class_weight': ["balanced", "weighted"],'penalty': ["l2", "l1"],'shuffle': [True, False],'random_state': [5, 10, 20]}

        self.rfc = RandomForestClassifier(class_weight="balanced", random_state=5)
        rf_grid = {"max_depth": [10, 15, 25],"criterion": ['gini', 'entropy'],"min_samples_split": [2, 3, 4, 5],"n_estimators": [10]}

        self.pac = PassiveAggressiveClassifier(class_weight="balanced")
        pac_grid = {'random_state': [5, 10, 15, 20],'C': [0.25, 0.5, 0.75, 1],'shuffle': [True, False]}

        self.rscv = RandomizedSearchCV(self.pac, pac_grid, random_state=20, n_jobs=-1)
        self.rscv.fit(XX, y)

        return

    def predict(self, X):

        X = self.preprocess(X)
        XX = self.preprocessor.transform(X["description"])
        predictions = self.rscv.predict(XX)
        return predictions