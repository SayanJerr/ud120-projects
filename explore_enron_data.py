#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
# from ud120-projects-master.final_project.poi_email_addresses import poiEmails
import pandas as pd
import numpy as np
import sys
sys.path.append("/Users/jerry/PycharmProjects/sayan_3/ud120-projects-master/final_project")
import poi_email_addresses

enron_data = pickle.load(open("/Users/jerry/PycharmProjects/sayan_3/ud120-projects-master/final_project/final_project_dataset.pkl", "rb"))
df = pd.DataFrame(enron_data)
# g =df.count(axis = 'index')
# # print(df)
# # timezone = []
# # for i in enron_data:
# #     timezone.append(enron_data[i]["poi"])
# # def get_counts(sequence):
# #     counts = {}
# #     for i in sequence:
# #         if i in counts:
# #             counts[i]+=1
# #         else :
# #             counts[i] = 1
# #     return counts

# # x = get_counts(timezone)
# # print(x)
df_count = df["total_payments"].value_counts()

