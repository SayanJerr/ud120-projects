#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

def classify(features_train, labels_train):
    from sklearn.tree import DecisionTreeClassifier
    clf= DecisionTreeClassifier()
    clf.fit(features_train,labels_train)
    return clf

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  


from skleearn.metrics import precision_score, recall_score

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.3, random_state = 42)

### it's all yours from here forward!  
from sklearn.tree import DecisionTreeClassifier
clf= DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print(precision_score(true_labels,predictions), recall_score(true_labels, predictions))