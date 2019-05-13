#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:57:45 2019

@author: saror
"""

def max_subarray_brute_force(arr):  
    """
    Brute-force Pseudo-code
    1. Find all subarrays
    2. Compute sum of each subarray
    3. Output the maximum sum
    complexity : O(n^2)
    """ 
    subarrays =[]
    sums = []
    for i in range(0, len(arr)):
        for j in range (i, len(arr)):
            subarray = arr [i : j+1]
            subarrays.append(subarrays)
            sums.append(sum(subarray))
         
    max_sum = max(sums)
    #max_idx = sums.index(max_sum)
        
    return max_sum
        

def max_subarray_crossing(arr, low, mid, high):
     left_sum = -10000000
     sum1 = 0
     for i in range(mid, low-1, -1):
         sum1 = sum1 + arr[i]
         if sum1 > left_sum:
             left_sum = sum1
             
     right_sum = -10000000
     sum1 = 0
     for i in range(mid+1, high+1):
         sum1 = sum1 + arr[i]
         if sum1 > right_sum:
             right_sum = sum1

     return left_sum + right_sum
    
      
def max_subarray_divide_conquer(arr, low, high):
    """
     Divide and Conquer Pseudo-code
         1. Divide the array into two parts - left and rigth
         2. Find the maximum subarray sum in the left part
         3. Find the maximum subarray sum in the right part
         4. Find the maximum subarray sum crossing the two parts containing the middle element
         5. Return the maximum of above 3 sums
         complexity : O(nlogn)
    """
    if (high == low): #only one element in array
        return arr[high]
    
    mid = (low+high)//2
    
    max_subarray_leftpart = max_subarray_divide_conquer(arr, low, mid)
    max_subarray_rightpart = max_subarray_divide_conquer(arr, mid+1, high)
    max_crossing = max_subarray_crossing(arr, low, mid, high)
    
    return max(max_subarray_leftpart, max_subarray_rightpart, max_crossing)

#testing
arr1 = [-2, -3, 1, 4, 5] 
arr2 = [-2,1,-3,4,-1,2,1,-5,4]
n1 = len(arr1)
n2 = len(arr2)

print("max_subarray_sum:", max_subarray_divide_conquer(arr1, 0, n1-1))
print("max_subarray_sum:", max_subarray_divide_conquer(arr2, 0, n2-1))   
