#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: shreya
"""
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

# array implementation of stack
class Stack_arr:
    def __init__(self):
        self.array = []
        
    def isEmpty(self):
        return len(self.array) == 0
        
     
    def peek(self):
        return self.array[len(self.C)-1]
    
    def push(self, data):
       # if self.isFull():
          #  print ("Stack is full")
          #  return
        self.array.append(data)
        
    def pop(self):
        if self.isEmpty():
            print("Stack is empty")
            return
        data = self.array.pop()
        return data
        
class Stack_ll:
    def __init__(self):
        self.top = None
    
    def isEmpty(self):
        return self.top == None

    def push(self, node):  #addBeg
        newNode = node
        newNode.next = self.top
        self.top = newNode
        
    def pop(self):  #deleteBeg
        if self.isEmpty():
            print ('Stack is empty')
            return
        
        temp = self.top
        self.top = self.top.next 
        popped_data = temp.data
        return popped_data
        
if __name__ == "__main__":
    stackMax = 10
    stack = Stack_arr()
    
    # pushing
    for i in range(0,10):
        stack.push(i)
    stack.push(10)
    print(stack.array)
    
    # popping
    print(stack.pop())
    print(stack.array)
    empty_stack = Stack_arr()
    print(empty_stack.pop())
    
    stack = Stack_ll()
    stack.push(Node(10))
    stack.push(Node(20))
    stack.push(Node(30))
    
    print(stack.pop())
    
#Stack_ll has no overflow problem 