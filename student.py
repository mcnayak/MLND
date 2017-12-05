#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:10:05 2017

@author: mnayak
"""

import pandas as pd
from time import time
from IPython.display import display
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import visuals as vs


student_data = pd.read_csv("student-data.csv")
student_data.plot(kind='density',subplots=True,layout=(4,4),sharex=False)
n_students = len(student_data.index)
print "Number Of Student Records {}".format(n_students) 
n_features = len(student_data.columns)
print "Number of Features {}".format(n_features)

n_passed = 0
for _,student in student_data.iterrows():
        if student['passed'] == 'yes':
            n_passed = n_passed + 1

print "Number of Students Passed {}".format(n_passed)
print "Number of Students Failed {}".format(n_students - n_passed)

feature_cols  = list(student_data.columns[:-1])
target_col = student_data.columns[-1]
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)


X_all = student_data[feature_cols]
y_all = student_data[target_col]

print "Number of Rows {}".format(len(X_all.index))

def preprocess_features(X):
    
    output = pd.DataFrame(index = X.index)
    
    for col,col_data in X.iterrows():
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        if col_data.dtype == object:
             col_data = pd.get_dummies(col_data, prefix = col)  
    output = output.join(col_data)
    return output

preprocess_features(student_data)
print student_data.head(5)

def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))

# TODO: Import any additional functionality you may need here
from sklearn.model_selection import train_test_split
# TODO: Set the number of training points
num_train = 320

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above
X_train = None
X_test = None
y_train = None
y_test = None
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=42)
# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))
    
    
    
    
# TODO: Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import linear_model 
# found KNN, SVM, SGDClassifier
# from sklearn import model_C

# TODO: Initialize the three models
clf_A = GaussianNB()
clf_B = DecisionTreeClassifier()
clf_C = KNeighborsClassifier()
clf_D = svm.SVC()
clf_E = linear_model.LogisticRegression()
clf_F = linear_model.SGDClassifier()

# TODO: Set up the training set sizes
X_train_100 = X_train[0:100]
y_train_100 = y_train[0:100]

X_train_200 = X_train[0:200]
y_train_200 = y_train[0:200]

X_train_300 = X_train[0:320]
y_train_300 = y_train[0:320]

# TODO: Execute the 'train_predict' function for each classifier and each training set size
#train_predict(clf_B, X_train_200, y_train_200, X_test, y_test)    
    
results = {}
for clf in [clf_A, clf_B, clf_C,clf_D,clf_E,clf_F]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    train_predict(clf,X_train_300,y_train_300,X_test,y_test)
    
#    for i, samples in enumerate([X_train_100, X_train_200, X_train_300]):
#         results[clf_name][i] = \
#         train_predict(clf, samples, y_train, X_test, y_test)

#for clf in [clf_A, clf_B, clf_C]:
#    clf_name = clf.__class__.__name__
#    print(results[clf_name])
    
# loading library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train_300, y_train_300)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
print accuracy_score(y_test, pred)

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, n_features=7,centers=2,
                  random_state=0, cluster_std=1.0)
#plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow');
print(X.shape)
print(y.shape)

#from sklearn.metrics import make_scorer, r2_score

#from sklearn.ensemble import AdaBoostClassifier

# TODO: Import a supervised learning model that has 'feature_importances_'
#clf = AdaBoostClassifier(random_state=543)

# TODO: Train the supervised model on the training set using .fit(X_train, y_train)
#model = clf.fit(X_train_300,y_train_300)


# TODO: Extract the feature importances using .feature_importances_ 
#importances = clf.feature_importances_


# TODO: Import 'GridSearchCV' and 'make_scorer'
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import make_scorer, r2_score

# TODO: Create the parameters list you wish to tune
#parameters = None

# TODO: Initialize the classifier
#clf = None

# TODO: Make an f1 scoring function using 'make_scorer' 
#f1_scorer = None

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
#grid_obj = None

# TODO: Fit the grid search object to the training data and find the optimal parameters
#grid_obj = None

# Get the estimator
#clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
#print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
#print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))