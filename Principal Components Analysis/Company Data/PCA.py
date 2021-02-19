# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:43:52 2021

@author: Anirudha Mulgund
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np

data=pd.read_csv("CompanyData.csv")

#Applying Logistic regression
df=data.drop(['Co_Code','Type'],axis=1)

X=df.values
y=data['Type'].values

#Spliting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=100)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


#PCA (Principal components analysis)

from sklearn.decomposition import PCA
pca=PCA(n_components=4)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=np.round(pca.explained_variance_ratio_,4)
explained_variance


#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)

y_pred=log_reg.predict(X_test)

print("Training Accuracy:{:.3f}".format(log_reg.score(X_train,y_train)))

print('Testing Accuracy:{:.3f}'.format(log_reg.score(X_test,y_test)))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#Cross Validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(log_reg,X,y,cv=10)
#accuracies=cross_val_score(log_reg,X,y,cv=5)
print('{:.3f}'.format(accuracies.mean())) #This validation is without the standardised values i.e. it is on the original values that we don't want

#Pipeline (This is used to make space for standardization and log_reg togather in one variable and use it togather on validation)
from sklearn.pipeline import make_pipeline
clf=make_pipeline(sc,log_reg) # Variable(clf) assigned to store sc(standard scale) and log_reg (logarithmic regression) function
accuracies=cross_val_score(clf,X,y,cv=10)
print('{:.3f}'.format(accuracies.mean()))
