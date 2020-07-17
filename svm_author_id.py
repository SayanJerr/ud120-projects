#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
import sys
from time import time
sys.path.insert(1,"/Users/jerry/PycharmProjects/sayan_3/ud120-projects-master/tools/")
from email_preprocess import preprocess



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

m = round(len(features_train)/100)
n = round(len(labels_train)/100)
features_train = features_train[:m]
labels_train = labels_train[:n]


#########################################################
### your code goes here ###
def NBAccuracy(features_train, labels_train, features_test, labels_test):   
    ## import the sklearn module for GaussianNB
    ## create classifier
    ## fit the classifier on the training features and labels
    ## return the fit classifier
    
    
    ## your code goes here!
    from sklearn import svm
    clf = svm.SVC(C= 10000,kernel = 'linear')
    t0 = time()
    clf.fit(features_train,labels_train)
    # # your clf.fit() line of code >
    print ("training time:", round(time()-t0, 3), "s"  )
    t0 = time()
    pred = clf.predict(features_test)
    # # your clf.fit() line of code >
    print( "predicting time:", round(time()-t0, 3), "s")
    # count = 0
    # for i in range(len(features_test)):
    #     if pred[i] == 1:
    #         count += 1
    #     else:
    #         pass
    # print(count)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    return acc

#########################################################
print(NBAccuracy(features_train, labels_train, features_test, labels_test))

#########################################################


