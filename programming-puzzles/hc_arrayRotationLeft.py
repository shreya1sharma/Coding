#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 23:18:25 2018

@author: shreya
"""

import math
import os
import random
import re
import sys

# Complete the rotLeft function below.
def rotLeft(a, d):
    n = len(a)
    b = list(a)
    for i in range(0, n):
        idx = n-d+i
        if idx < n:
            b[idx] = a[i]
        else:
            b[idx-n] = a[i] 
    return b

if __name__ == '__main__':
    

    nd = input().split()

    n = int(nd[0])

    d = int(nd[1])

    a = list(map(int, input().rstrip().split()))

    result = rotLeft(a, d)

'''
Example input
5 4
1 2 3 4 5
'''