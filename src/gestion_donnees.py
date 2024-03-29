# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Si sur mac : train_path='../data/train.csv'
def open_data(train_path='../data/train.csv'):
    train_df = pd.read_csv(train_path)
    return train_df


def showdown_df():
    # Logging for Visual Comparison
    log_cols = ["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)
    return log


def showdownPutter(name, acc, ll):
    log_cols = ["Classifier", "Accuracy", "Log Loss"]
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    return log_entry


def display_scores(scores):
    score_df = pd.DataFrame({'Scores': scores})
    score_df['Mean'] = score_df['Scores'].mean()
    score_df['Std Variation'] = score_df['Scores'].std()
    return score_df


class GestionDonnees:

    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None, labels=None, classes=None, train_df=None, test_df=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.labels = labels
        self.classes = classes
        self.train_df = open_data()

    def prepocess(self):
        coder = LabelEncoder().fit(self.train_df.etiquette)
        self.labels = coder.transform(self.train_df.etiquette)
        self.classes = list(coder.classes_)

        self.train_df = self.train_df.drop(['etiquette'], axis=1)
        

    def stratifiedSelection(self):
        # split = StratifiedShuffleSplit(
        #     n_splits=10, test_size=0.2, random_state=42)

        # for train_index, test_index in split.split(self.train_df, self.labels):
        #     self.x_train, self.x_test = self.train_df.values[
        #         train_index], self.train_df.values[test_index]
        #     self.y_train, self.y_test = self.labels[train_index], self.labels[test_index]

        self.x_train, self.x_test, self.y_train, self.y_test=train_test_split(self.train_df,self.labels,test_size=0.20,shuffle=True)
        self.normalize()

    def normalize(self):
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.fit_transform(self.x_test)
