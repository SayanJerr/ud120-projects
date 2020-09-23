# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 08:36:36 2020

@author: jerry
"""

def isNaN(num):
    if num == num:
        return num
    elif num == 'NaN' :
        return 0
    else:
        return 0
def get_details(feature,data_dict):
    m = []
    details = []
    m.clear()
    details.clear()
    keys = data_dict.keys()
    m = [data_dict[k][feature] for k in keys]
    for i in range(0,len(m)):
        if m[i]=="NaN":
            m[i]=0
    for i,j in zip(details,data_dict):
        data_dict[j][feature] = i
    return data_dict
        
def replace_details(m, j, feature,data_dict):
        m=0
        for i in data_dict:
            while m <= len(m):
              data_dict[i][feature][m] = j


def select_clf(n_features, features, labels):   

    from sklearn.model_selection import StratifiedShuffleSplit
    import sklearn.metrics as metrics
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    #from operator import itemgetter
    clf_list = []
    select_k = SelectKBest(f_classif, k=n_features)
    features = select_k.fit_transform(features, labels)
    scores = select_k.scores_
    features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.3,random_state=42)
    
    dt = DecisionTreeClassifier()
    parameters = {'min_samples_split': [2, 3, 4, 5, 6, 7],
                  'max_features': ['auto', 'sqrt', 'log2', None],
                  'criterion': ['gini', 'entropy']}

    clf_dt = GridSearchCV(dt, parameters, scoring='f1')

    rf = RandomForestClassifier()
    parameters = {'min_samples_split': [2, 3, 4, 5, 6, 7],
                  'max_features': ['auto', 'sqrt', 'log2', None],
                  'criterion': ['gini', 'entropy'],
                  'n_estimators': [2, 3, 4, 5, 6, 7]}
    clf_rf = GridSearchCV(rf, parameters, scoring='f1')

    knn = KNeighborsClassifier()
    parameters = {'n_neighbors': [1, 3, 5, 7, 9]}
    clf_knn = GridSearchCV(knn, parameters, scoring='f1')

    svm = SVC()
    parameters = {'kernel': ['rbf'],
                  'C': [1, 10, 100, 1000, 10000, 100000]}
    clf_svm = GridSearchCV(svm, parameters, scoring='f1')


    clf_dt.fit(features_train, labels_train)
    clf_dt = clf_dt.best_estimator_
    pred_dt = clf_dt.predict(features_test)
    # print '\n\nDecision Tree'
    # print 'Accuracy: ', metrics.accuracy_score(pred_dt, labels_test)
    #recall_dt = metrics.recall_score(pred_dt, labels_test)
    #precision_dt = metrics.precision_score(pred_dt, labels_test)
    # print 'F1 score: ', metrics.f1_score(pred_dt, labels_test)
    print (clf_dt)
    clf_list.append([metrics.f1_score(pred_dt, labels_test), metrics.accuracy_score(pred_dt, labels_test), n_features, scores, clf_dt])

    clf_rf.fit(features_train, labels_train)
    clf_rf = clf_rf.best_estimator_
    pred_rf = clf_rf.predict(features_test)
    # print '\n\nRandon Forest'
    # print 'Accuracy: ', metrics.accuracy_score(pred_rf, labels_test)
    # print 'Recall: ', metrics.recall_score(pred_rf, labels_test)
    # print 'Precision: ', metrics.precision_score(pred_rf, labels_test)
    # print 'F1 score: ', metrics.f1_score(pred_rf, labels_test)
    print (clf_rf)
    clf_list.append([metrics.f1_score(pred_rf, labels_test), metrics.accuracy_score(pred_rf, labels_test), n_features, scores, clf_rf])

    clf_knn.fit(features_train, labels_train)
    clf_knn = clf_knn.best_estimator_
    pred_knn = clf_knn.predict(features_test)
    # print '\n\nKNN'
    # print 'Accuracy: ', metrics.accuracy_score(pred_knn, labels_test)
    # print 'Recall: ', metrics.recall_score(pred_knn, labels_test)
    # print 'Precision: ', metrics.precision_score(pred_knn, labels_test)
    # print 'F1 score: ', metrics.f1_score(pred_knn, labels_test)
    print (clf_knn )
    clf_list.append([metrics.f1_score(pred_knn, labels_test), metrics.accuracy_score(pred_knn, labels_test), n_features, scores, clf_knn])

    clf_svm.fit(features_train, labels_train)
    clf_svm = clf_svm.best_estimator_
    pred_svm = clf_svm.predict(features_test)
    # print '\nSVM'
    # print 'Accuracy: ', metrics.accuracy_score(pred_svm, labels_test)
    # print 'Recall: ', metrics.recall_score(pred_svm, labels_test)
    # print 'Precision: ', metrics.precision_score(pred_svm, labels_test)
    # print 'F1 score: ', metrics.f1_score(pred_svm, labels_test)
    print (clf_svm)
    clf_list.append([metrics.f1_score(pred_svm, labels_test), metrics.accuracy_score(pred_svm, labels_test), n_features, scores, clf_svm])

    order_clf_list = sorted(clf_list, key=lambda x: x[0] )
    return order_clf_list

#from operator import itemgetter
#clf_list = []
#for k in range (1, int(len(features_list) / 2)): # Try sets of 1 - number_of_features / 2
    #clf_list.append(select_clf(k, features, labels))
#order_clf_list = sorted(clf_list, key=itemgetter(0, 1)) # order by f1-score and accuracy
#clf = order_clf_list[len(order_clf_list) - 1][4]
def clf_and_features(order_clf_list,features_list):
    from operator import itemgetter
    i=0
    c_list = []
    new_features_list = []
    while i < len(order_clf_list) :
        clf = order_clf_list[i][4]
        print ('\n\nClf: ', clf)
        c_list.append(clf)
        number_of_features = order_clf_list[i][2]
        print ('\n\nNumber of features: ', number_of_features)

        print ('\n\nFeatures and scores: ')
        score_list = order_clf_list[i][3]
        features = features_list[1:]
        features_scores = []
        m = 0
        for feature in features:
            features_scores.append([feature, score_list[m]])
            m += 1
        features_scores = sorted(features_scores, key=itemgetter(1))
        print (features_scores[::-1])

        print ('\n\nFeatures used: ')
        nnew_features_list = []
        for feature in features_scores[::-1][:number_of_features]:
            nnew_features_list.append(feature[0])
        print (nnew_features_list)
        print ('\n\n')
        nnew_features_list = ['poi'] + nnew_features_list
        new_features_list.append(nnew_features_list)
        i +=1
    

   
       
    return new_features_list,c_list