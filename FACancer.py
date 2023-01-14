# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 13:37:02 2022

@author: ivana
"""

import sys
sys.path.append('C:\\Users\\ivana\\Documents\\Faks\\5. godina\\III semestar\\Projekt II\\Kod\\Ivana')
work_dir = 'C:\\Users\\ivana\\Documents\\Faks\\5. godina\\III semestar\\Projekt II\\Kod\\Ivana\\FACancer\\Logger\\'

from niapy.algorithms.basic import FireflyAlgorithm
from niapy.task import Task
from niapy.problems import Problem

from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
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
        accuracy = cross_val_score(SVC(), self.X_train[:, selected], self.y_train, cv=2, n_jobs=-1).mean()
        score = 1 - accuracy
        num_features = self.X_train.shape[1]
        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)
    

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
log_file = open(str(work_dir + 'logger_cancer_FA ' + datetime.now().strftime('%Y-%m-%d %H-%M-%S') + '.txt'), 'w')
sys.stdout = log_file
print('Firefly Algorithm')
print()

lst_accu_stratified = []
lst_f1score_stratified = []
time_log = 0
for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    x_train_fold, x_test_fold = x[train_index], x[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    
    task = Task(FeatureSelection(x_train_fold, y_train_fold), max_iters=5)
    
    algorithm = FireflyAlgorithm()
    
    beginning = time.time()
    best_features, best_fitness = algorithm.run(task)
    end = time.time()
    selected = best_features > 0.5
    
    print(f"Number of selected features {i+1}.iter: ", selected.sum(), " from ", len(best_features))

    X_train_fold_fs = x_train_fold[:, selected]
    X_test_fold_fs = x_test_fold[:, selected]

    clf.fit(X_train_fold_fs,y_train_fold)
    lst_accu_stratified.append(clf.score(X_test_fold_fs,y_test_fold))
    lst_f1score_stratified.append(f1_score(y_test_fold, clf.predict(X_test_fold_fs), average="weighted"))
    time_log += (end - beginning)
    ConfusionMatrixDisplay.from_estimator(clf, X_test_fold_fs, y_test_fold)
    plt.savefig('C:\\Users\\ivana\\Documents\\Faks\\5. godina\\III semestar\\Projekt II\\Kod\\Ivana\\FACancer\\Matrix\\default\\FACancer' + str(i) +'.png')
    plt.show()
    
print('\nTime: ', str(time_log), ' s')
print()
print('List of f1 score: ', lst_f1score_stratified)
print('Overall f1_score: ', np.mean(lst_f1score_stratified))
print('Standard Deviation is: ', np.std(lst_f1score_stratified))
print()
print('List of possible accuracy:', lst_accu_stratified)
print('Overall Accuracy: ', np.mean(lst_accu_stratified))
print('Standard Deviation is: ', np.std(lst_accu_stratified))

sys.stdout = old_stdout
log_file.close()