'''
Method 1: recursion

pseudo-code
1. read the first element
2. initialize a canJump variable to False
3. Check all possible jumps starting from maximum to 0 (reverse order)
4. For each jump repeat steps 1-3
5. if the jump leads to the end of the array (len(arr)=1):
 Update canJump to True, else do not update
6. If while jumping the arr length is exceeded:
    reduce the jump size and repeat steps 1-5
7. return canJump
'''
def jumps(arr):
    
    first_element = arr[0]
    canJump = False
    
    if len(arr)==1:
          return True
            
    for i in range( first_element, 0, -1):
        if len(arr[i:])!=0:
            canJump = jumps(arr[i:])
        
        if canJump == True:
            break
    
    return canJump
   

'''
Method 2: dynamic programming

pseudo-code

'''

#%%   