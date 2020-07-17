#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
import matplotlib as plt
from time import time
sys.path.insert(1,"/Users/jerry/PycharmProjects/sayan_3/ud120-projects-master/tools/")
sys.path.insert(2,"/Users/jerry/PycharmProjects/sayan_3/ud120-projects-master/choose_your_own")
from email_preprocess import preprocess
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


features_train, labels_train, features_test, labels_test = makeTerrainData()

#########################################################
### your code goes here ###
def NBAccuracy(features_train, labels_train, features_test, labels_test):   
    ## import the sklearn module for GaussianNB
    ## create classifier
    ## fit the classifier on the training features and labels
    ## return the fit classifier
    
    
    ## your code goes here!
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    # t0 = time()
    clf.fit(features_train,labels_train)
    # # your clf.fit() line of code >
    # print ("training time:", round(time()-t0, 3), "s"  )
    # t0 = time()
    pred = clf.predict(features_test)
    # # your clf.fit() line of code >
    # print( "predicting time:", round(time()-t0, 3), "s")
   
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    return acc

#########################################################
clf = NBAccuracy(features_train, labels_train, features_test, labels_test)
prettyPicture(clf, features_test, labels_test)
plt.show()
