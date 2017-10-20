# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns].to_dict(orient='record')

if __name__ == '__main__':

    feature_dict = {'delete': ['PassengerId', 'Survived', 'Name', 'Ticket'],
                    'quant_list': ['Age', 'Fare', 'SibSp', 'Parch'],
                    'categ_list': ['Pclass', 'Sex', 'Cabin', 'Embarked']}

    pipeline = Pipeline([
        ('union', FeatureUnion([
            ('quant', Pipeline([
                ('extract', ColumnSelector(feature_dict['quant_list'])),
                ('dicVect', DictVectorizer(sparse=False)),
                ('scaler', StandardScaler()),

            ])),
            ('categ', Pipeline([
                ('extract', ColumnSelector(feature_dict['categ_list'])),
                ('dicVect', DictVectorizer()),

            ])),

        ])),
        ('nor', Normalizer()),

    ])

    # read data
    train = pd.read_csv('train.csv')
    train.dropna(inplace=True)

    # get X, y
    y = train.Survived
    X = train.drop(feature_dict['delete'], axis=1)

    # split to train data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # transform the feature
    X_train = pipeline.fit_transform(X_train)

    # fit the classifier
    clf = LogisticRegression(class_weight="balanced")
    clf.fit(X_train, y_train)

    # transform the test feature
    X_test = pipeline.transform(X_test)

    # use classifier
    y_pred = clf.predict(X_test)

    # print the result
    print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))
    print('accuracy: {0:.2f}'.format(accuracy_score(y_test, y_pred)))
