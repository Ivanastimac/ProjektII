# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 18:14:40 2022

@author: ivana
"""

import sys
sys.path.append('C:\\Users\\ivana\\Documents\\Faks\\5. godina\\III semestar\\Projekt II\\Kod\\Ivana')
work_dir = 'C:\\Users\\ivana\\Documents\\Faks\\5. godina\\III semestar\\Projekt II\\Kod\\Ivana\\FPACancer\\Logger\\'

import pyRAPL

from niapy.algorithms.basic import FlowerPollinationAlgorithm
from niapy.task import Task
from niapy.problems import Problem

from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import time
import math

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold


class FeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.99):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha

    def _evaluate(self, x):
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        f1 = make_scorer(f1_score, average='weighted')
        fitness = cross_val_score(DecisionTreeClassifier(), self.X_train[:, selected], self.y_train, cv=2, n_jobs=-1, scoring=f1).mean()
        score = 1 - fitness
        num_features = self.X_train.shape[1]
        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)
    
pyRAPL.setup()
meter1 = pyRAPL.Measurement('bar')
meter2 = pyRAPL.Measurement('bar')

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

x = dataset.data
y = dataset.target
feature_names = dataset.feature_names

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(x, y)
StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

clf = RandomForestClassifier(n_estimators=10, random_state=42)

old_stdout = sys.stdout
log_file = open(str(work_dir + 'logger_cancer_FPA ' + datetime.now().strftime('%Y-%m-%d %H-%M-%S') + '.txt'), 'w')
sys.stdout = log_file
print('Flower Pollination Algorithm (f1 score for fitness function)')
print()

lst_accu_stratified = []
lst_f1score_stratified = []
lst_auc_roc = []
#lst_geo_mean = []
#time_log = 0
for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    x_train_fold, x_test_fold = x[train_index], x[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    
    task = Task(FeatureSelection(x_train_fold, y_train_fold), max_iters=5)
    
    algorithm = FlowerPollinationAlgorithm()
    
    meter1.begin()
    #beginning = time.time()
    best_features, best_fitness = algorithm.run(task)
    #end = time.time()
    meter1.end()
    
    selected = best_features > 0.5
    
    print(f"Number of selected features {i+1}.iter: ", selected.sum(), " from ", len(best_features))

    X_train_fold_fs = x_train_fold[:, selected]
    X_test_fold_fs = x_test_fold[:, selected]

    meter2.begin()
    clf.fit(X_train_fold_fs,y_train_fold)
    meter2.end()
    
    lst_accu_stratified.append(clf.score(X_test_fold_fs,y_test_fold))
    lst_f1score_stratified.append(f1_score(y_test_fold, clf.predict(X_test_fold_fs), average="weighted"))
    lst_auc_roc.append(roc_auc_score(y_test_fold, clf.predict_proba(X_test_fold_fs)[:, 1]))
    #time_log += (end - beginning)
    #cm = confusion_matrix(y_test_fold, clf.predict(X_test_fold_fs))
    #lst_geo_mean.append(math.sqrt((cm[0, 0] / (cm[0, 0] + cm[1, 0])) * ))
    ConfusionMatrixDisplay.from_estimator(clf, X_test_fold_fs, y_test_fold)
    plt.savefig('C:\\Users\\ivana\\Documents\\Faks\\5. godina\\III semestar\\Projekt II\\Kod\\Ivana\\FPACancer\\Matrix\\f1\\FPACancer' + str(i) +'.png')
    plt.show()
    print("\nIteracija " + str(i) + " fs-a \n" + str(meter1.result) + "\n")
    print("\nIteracija " + str(i) + " clf-a \n" + str(meter2.result) + "\n")
    
#print('\nTime: ', str(time_log), ' s')
#print()
print('List of f1 score: ', lst_f1score_stratified)
print('Overall f1_score: ', np.mean(lst_f1score_stratified))
print('Standard Deviation is: ', np.std(lst_f1score_stratified))
print()
print('List of possible accuracy:', lst_accu_stratified)
print('Overall Accuracy: ', np.mean(lst_accu_stratified))
print('Standard Deviation is: ', np.std(lst_accu_stratified))
print()
print('List of possible auc_roc: ', lst_auc_roc)
print('Overall auc_roc: ', np.mean(lst_auc_roc))
print('Standard Deviation is: ', np.std(lst_auc_roc))

sys.stdout = old_stdout
log_file.close()