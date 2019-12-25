# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:53:49 2018

@author: hoge
"""

#2
def factorial(a):
    if a==0:
        return 1
    else:
        return a*factorial(a-1)    
a= int(input())
print(factorial(a))

#%%3
a= int(input())
d={}
for i in range(1,a+1):
    d[i]= i*i
print(d)

#%%4
a= input()
a=a.split(',')
print(a)
print(tuple(a))
#%%5
class stringobject(object):
    def __init__(self):
        self.s=""
    def getstring(self):
        self.s= input()
    def printstring(self):
        print(self.s.upper())
    
a= stringobject()
a.getstring()
a.printstring()
#%%
class computearea(object):
    def __init__(self):
        self.h=int
        self.w=int
    def getdim(self):
        self.h=int(input())
        self.w=int(input())
    def area(self):
        print(self.h*self.w)
        
a= computearea()
a.getdim()
a.area()