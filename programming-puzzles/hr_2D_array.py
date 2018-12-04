#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 23:13:11 2018

@author: shreya
"""

import math
import os
import random
import re
import sys


# Complete the hourglassSum function below.
def hourglassSum(arr):
    sums = []
    for i in range(0,4): # i is row
        for j in range(0,4): # j is col
            elms = []
            elms.append(arr[i][j:j+3])
            elms.append([arr[i+1][j+1]])
            elms.append(arr[i+2][j:j+3])
            sums.append(sum(sum(x) for x in elms))
    return max(sums)



if __name__ == '__main__':
    
    arr = []

    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))

    result = hourglassSum(arr)
'''
#Example input
1 0 0 0 0 0

1 0 0 0 0 0

1 0 0 0 0 0

1 0 0 0 0 0

1 0 0 0 0 0

1 0 0 0 0 0
'''