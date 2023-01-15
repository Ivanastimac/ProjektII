# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 13:40:32 2022

@author: ivana
"""

import sys
sys.path.append('C:\\Users\\ivana\\Documents\\Faks\\5. godina\\III semestar\\Projekt II\\Kod\\Ivana')
work_dir = 'C:\\Users\\ivana\\Documents\\Faks\\5. godina\\III semestar\\Projekt II\\Kod\\Ivana\\FAAds\\Logger\\'

import pyRAPL

from niapy.algorithms.basic import FireflyAlgorithm
from niapy.task import Task
from niapy.problems import Problem

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, make_scorer, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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
    
df = pd.read_csv('ad.data', delimiter=',', header=None, low_memory=False, )  
df_data = df.iloc[:,:-1]
df_class = df.iloc[:,-1]
df_data = df_data.replace('?', np.NaN)
df_data = df_data.replace('   ?', np.NaN)
df_data = df_data.replace('     ?', np.NaN)
thresh = len(df_data) * 0.4
df_data.dropna(thresh = thresh, axis = 1, inplace = True)
imp_mean = SimpleImputer(missing_values=np.NaN, strategy='median')
imputer = imp_mean.fit(df_data)
df_imp = imputer.transform(df_data)
df_data = pd.DataFrame(df_imp)

std_scaler = StandardScaler()
x_scaled = std_scaler.fit_transform(df_data.values) 
df_data = pd.DataFrame(x_scaled, index = df_data.index)

x = df_data
y = df_class

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(x.values, y.values)
StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

clf=RandomForestClassifier(n_estimators=10, random_state=42)

old_stdout = sys.stdout
log_file = open(str(work_dir + 'logger_ads_FA ' + datetime.now().strftime('%Y-%m-%d %H-%M-%S') + '.txt'), 'w')
sys.stdout = log_file
print('Firefly Algorithm (f1 score for fitness function)')
print()

lst_accu_stratified = []
lst_f1score_stratified = []
lst_auc_roc = []
#time_log = 0
for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    x_train_fold, x_test_fold = x.iloc[train_index], x.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    
    problem = FeatureSelection(x_train_fold.values, y_train_fold.values)
    task = Task(problem, max_iters=5)
    algorithm = FireflyAlgorithm()
        
    meter1.begin()
    #beginning = time.time()
    best_features, best_fitness = algorithm.run(task)
    #end = time.time()
    meter1.end()
    
    selected = best_features > 0.5

    print(f"Number of selected features {i+1}.iter: ", selected.sum(), " from ", len(best_features))
    
    X_train_fold_fs = x_train_fold.iloc[:, selected]
    X_test_fold_fs = x_test_fold.iloc[:, selected]
    
    meter2.begin()
    clf.fit(X_train_fold_fs,y_train_fold)
    meter2.end()
    
    lst_accu_stratified.append(clf.score(X_test_fold_fs,y_test_fold))
    lst_f1score_stratified.append(f1_score(y_test_fold, clf.predict(X_test_fold_fs), average="weighted"))
    lst_auc_roc.append(roc_auc_score(y_test_fold, clf.predict_proba(X_test_fold_fs)[:, 1]))
    #time_log += (end - beginning)
    ConfusionMatrixDisplay.from_estimator(clf, X_test_fold_fs, y_test_fold)
    plt.savefig('C:\\Users\\ivana\\Documents\\Faks\\5. godina\\III semestar\\Projekt II\\Kod\\Ivana\\FAAds\\Matrix\\f1\\FAAds' + str(i) +'.png')
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

