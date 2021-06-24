import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score , mean_absolute_error, mean_squared_error

df = pd.read_csv('Classification.csv')

# manually add intercept
#df['intercept'] = 1
independent_variables = ['Hours_Studied']#, 'intercept']
x = df[independent_variables] # independent variable
y = df['Result'] # dependent variable
# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(x, y)
 #check the accuracy on the training set
model.score(x, y)
from sklearn import metrics
# generate evaluation metrics
print ("Accuracy :", metrics.accuracy_score(y, model.predict(x)))
print ("AUC :", metrics.roc_auc_score(y, model.predict_proba(x)[:,1]))
print ("Confusion matrix :",metrics.confusion_matrix(y, model.predict(x)))
print ("classification report :", metrics.classification_report(y, model.predict(x)))
