#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.insert(0,"/Users/jerry/PycharmProjects/sayan_3/ud120-projects-master/tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load(open("/Users/jerry/PycharmProjects/sayan_3/ud120-projects-master/final_project/final_project_dataset.pkl", "rb"))
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


