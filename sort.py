#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def _swap(arr, i, j):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp
    return arr


#selection sort

'''
Pesudo code:
    
for i from 0 to n-1:
    Select the ith element
    Find the smallest value between i and n-1
    Swap the smallest value with the ith element
   
Complexity: O(n^2), sigma(n^2)
'''    
def selection_sort(arr):
    for i in range(0, len(arr)-1):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j]<arr[min_idx]:
                min_idx= j
        arr = _swap(arr, i, min_idx)
    return arr
   
def selection_sort_2(arr):
    for i in range(0, len(arr)-1):
        min_value = np.min(arr[i:])
        min_idx = arr.index(min_value)
        arr = _swap(arr, i, min_idx)
    return arr
        
#bubble sort
    
'''
Pseudo code:

Set swap counter to non-zero value
repeat until swap_counter is zero:
    Reset swap_counter to 0
    for i from 0 to n-2:
        if the elements ith and i+1th are out of order
            swap them
            add one to swap_counter
            
Complexity: O(n^2), sigma(n)
'''

def bubble_sort(arr):
    swap_counter=-1
    while(swap_counter>0):
        swap_counter=0
        for i in range(len(arr)-1):
            if arr[i]>arr[i+1]:
                arr = _swap(arr, i, i+1)
                swap_counter= swap_counter+1
    return arr

#insertion sort
'''
Psuedo code:
    
Call the first element as 'sorted'
Repeat until all elements are sorted:
    look at the next unsorted element. 
    Insert into the 'sorted' portion by shifting the required number of elements.

Complexity: O(n^2), sigma(n)
'''

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j>=0 and arr[j]>key:
            arr[j+1] = arr[j]
            j = j-1
        arr[j+1] = key
        print(arr)
    return arr
            
#merge sort
'''
Pseudo code:
on input of n elements 
    if n<2
        return
    else
        sort left half of elements
        sort right half of elements
        merge sorted halves 
        
Complexity : O(nlog n), sigma(nlog n)
'''

def merge_sort(arr):
    n = len(arr)
    if len(arr)>1:
        mid = n//2
        lefthalf = arr[:mid]
        righthalf = arr[mid:]
        merge_sort(lefthalf)
        merge_sort(righthalf)
        i = j = k = 0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i]<righthalf[j]:
                arr[k]=lefthalf[i]
                i=i+1
            else:
                arr[k]=righthalf[j]
                j=j+1
            k=k+1
        while i < len(lefthalf):
           arr[k] = lefthalf[i]
           i = i+1
           k = k+1
           
        while j < len(righthalf):
           arr[k] = righthalf[j]
           j = j+1
           k = k+1
    return arr
