#!/usr/bin/python

from my_functttt import select_clf
import sys
import numpy as np
import pickle
from sklearn import preprocessing
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,load_classifier_and_data,test_classifier
from my_functttt import clf_and_features
from operator import itemgetter
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, train_test_split,StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
from sklearn.ensemble import RandomForestClassifier
#from operator import itemgetter
#clf_list = []
#for k in range (1, int(len(features_list) / 2)): # Try sets of 1 - number_of_features / 2
    #clf_list.append(select_clf(k, features, labels))
#order_clf_list = sorted(clf_list, key=itemgetter(0, 1)) # order by f1-score and accuracy
#clf = order_clf_list[len(order_clf_list) - 1][4]
### Load the dictionary containing the dataset
with open("/Users/jerry/PycharmProjects/sayan_3/ud120-projects-master/final_project/final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

initial_features_list = data_dict['METTS MARK'].keys()

features_list = []
for i in  data_dict['METTS MARK'].keys():
    features_list.append(i)
features_list.remove('poi')
features_list.remove('email_address')
features_list =['poi'] + features_list

for name in data_dict:
        for feature in initial_features_list:
            if data_dict[name][feature] == 'NaN':
                data_dict[name][feature] = 0


### Task 2: Remove outliers 
identified_outliers = ["TOTAL","THE TRAVEL AGENCY IN THE PARK"]


print (("Original Length"), len(data_dict))
for outlier in identified_outliers:
    data_dict.pop(outlier)    

keys = data_dict.keys()


print (("Length after Outlier"), len(data_dict) )


for name in data_dict:
        for feature in initial_features_list:
            if data_dict[name][feature] == 'NaN':
                data_dict[name][feature] = 0

### Task 3: Create new feature(s)
for key in keys:
    if data_dict[key]['from_poi_to_this_person']!=0:
        data_dict[key]['percentage_from_poi']= float(data_dict[key]['from_poi_to_this_person'])/float(data_dict[key]['to_messages'])
    else:
        data_dict[key]['percentage_from_poi']=0
    if data_dict[key]['from_this_person_to_poi']==0:
        data_dict[key]['percentage_to_poi']=0
    else:
        data_dict[key]['percentage_to_poi']= float(data_dict[key]['from_this_person_to_poi'])/float(data_dict[key]['from_messages'])


features_list +=  ['percentage_to_poi', 'percentage_from_poi']


    
# my_dataset for easy export below.
my_dataset = data_dict
features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'percentage_to_poi', 'deferred_income', 'long_term_incentive', 'restricted_stock', 'total_payments', 'shared_receipt_with_poi']
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.3,random_state=42)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(criterion='entropy', max_features='log2')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

    
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.3,random_state=42)
print(clf)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = clf.score(features_test, labels_test)
print ("Accuracy: ", accuracy)
target_names = ['non_poi', 'poi']
print (classification_report(y_true = labels_test, y_pred =pred, target_names = target_names))
    
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


dump_classifier_and_data(clf, my_dataset, features_list)




