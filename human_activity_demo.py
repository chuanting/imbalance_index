# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: human_activity_demo.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-18 (YYYY-MM-DD)
-----------------------------------------------
"""
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
import pandas as pd

# folder is where you store the data
folder = 'D:/Projects/Github/HumanActivity/'
# first we load data and shuffle data
df_train = pd.read_csv(folder + 'train.csv')
df_train = df_train.sample(frac=1).reset_index(drop=True)
df_test = pd.read_csv(folder + 'test.csv')
df_test = df_test.sample(frac=1.).reset_index(drop=True)

# input features are all except the last two columns (the last one is label, the other one is personal id)
train_x = df_train.iloc[:, :-2]
test_x = df_test.iloc[:, :-2]
# the label is human activity, there are in total 6 types of activities
classes = df_train['Activity'].unique()
train_y = df_train['Activity']
test_y = df_test['Activity']

# we use a linear model called Logistic regression, it is the most classical ML model. It's a linear model!
model = LogisticRegression(max_iter=5000)

# model training, after this step, we obtain a trained model with the best parameters
model.fit(train_x, train_y)

# we use this model to perform prediction
pred = model.predict(test_x)

# we can get the prediction results by using the following metrics
print(metrics.classification_report(test_y, pred))
print(metrics.confusion_matrix(test_y, pred))
print(metrics.accuracy_score(test_y, pred))

lb = LabelBinarizer()
lb.fit(test_y)
y_test = lb.transform(test_y)
y_pred = lb.transform(pred)

print(metrics.roc_auc_score(y_test, y_pred))

