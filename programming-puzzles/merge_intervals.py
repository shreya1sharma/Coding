# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:03:25 2019

@author: 0000016446351
"""
import numpy as np

def merge_intervals(arr):
    n = len(arr)
    print(n)
    merged = []
    for l in range(0, n-1):
        first_interval = arr[l]
        second_interval = arr[l+1]

        if first_interval[1] >= second_interval[0]:
            sorted_interval = np.sort([first_interval[0], first_interval[1], second_interval[0], second_interval[1]])
            print(sorted_interval)
            sorted_interval = [sorted_interval[0], sorted_interval[3]]
            merged.append(sorted_interval)
            l = l+1
        else:
            merged.append(arr[l])
            merged.append(arr[l+1])
        
    return merged
        
        #%%
        