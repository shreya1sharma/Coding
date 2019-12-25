# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:49:21 2018

@author: hoge
"""

#list comprehension v/s generator
#Exaple 1
xyz= (i for i in range(4000000))  #generator object
print(list(xyz)[:5])

xyz= [i for i in range(4000000)]    #new list stored in memory
print(xyz[:5])

#example 2
input_list = [5,10,3,5,55]
def div_by_five(num):
    if num %5==0:
        return True
    else:
        return False
xyz= (i for i in input_list if div_by_five(i))
print(list(xyz))
xyz= [i for i in input_list if div_by_five(i)]
print(xyz)

#embedded list comprehension
for i in range(5):
    for j in range(3):
        print(i,j)
        
[[print(i,j) for j in range(3)] for i in range(i)]