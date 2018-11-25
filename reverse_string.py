#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 18:07:43 2018

@author: shreya
"""

from stack import Stack_arr

#using stack
def ReverseString(arr):
    length = len(arr)
    reverse = []
    s = Stack_arr()
    for i in range(0, length):
        s.push(arr[i])
    for i in range(0, length):
        reverse.append(s.pop())
    
    return reverse
        

arr = 'hello'
reverse = ReverseString(arr)
print(''.join(reverse))


#another simple approach

i = 0
j = len(arr)-1
arr = list(arr)
while(i<=j):
    a = arr[i]
    arr[i]=arr[j]
    arr[j]=a 
    i+=1
    j-=1
    
print(''.join(arr))
