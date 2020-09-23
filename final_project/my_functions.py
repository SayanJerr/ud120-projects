# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 08:36:36 2020

@author: jerry
"""

def get_details_nonan(feature,data_dict,**kwargs):
        details = []
        details.clear()
        for i in data_dict:
            details.append([data_dict[i][feature]])
        for i in range(len(details) - 1):
            if details[i] == 'NaN':
                details[i] = 0  
        return details
def replace_details(m, j, feature,data_dict):
        m=0
        for i in data_dict:
            while m <= len(m):
              data_dict[i][feature][m] = j