#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 11:50:59 2018

@author: shreya
"""

class Node:
    def __init__(self,data): #self is used for object
        self.data = data
        self.next = None
        
    def set_data(self, data):
        self.data = data
        
    def get_data(self):
        return self.data
    
    def set_next(self, data):
        self.next = next
        
    def get_next(self):
        return self.next
    
    def has_next(self):
        return self.next != None
    
    
class LinkedList(object):
    def __init__(self):
        self.length = 0
        self.head = None
        
    # method to add a node in linked list
    def addNode(self, node):
        if self.length == 0:
           self.addBeg(node)
        else:
            self.addLast(node)
       
    # method to add node at the beginning
    def addBeg(self, node):
        newNode = node
        newNode.next = self.head
        self.head = newNode
        self.length += 1
      
    # method to add node at the last
    def addLast(self, node):
        currentnode = self.head
        
        while currentnode.next != None:
            currentnode = currentnode.next
            
        newNode = node
        newNode.next = None
        currentnode.next = newNode
        self.length += 1
        
    # method to add node at a particular position
    def addAtPos(self, pos, node):
        count = 0
        currentnode = self.head
        previousnode = self.head
        
        if pos>self.length or pos<0:
            print("The position does not exist.")
         
        elif pos == 1:
            self.addBeg(node)
            self.length += 1
        
        else:
            while currentnode.next != None or count<pos:
                count = count + 1
                if count == pos:
                    previousnode.next = node
                    node.next = currentnode
                    self.length += 1
                    return   # causes the function to exit or terminate immediately, even if it is not the last statement of the function.
   
                else:
                    previousnode = currentnode
                    currentnode = currentnode.next
                    
    # method to add node after a particular value
    def addAftervalue(self, data, node):
        newNode = node
        currentnode = self.head
        
        while currentnode.next != None or currentnode.data != data:
            currentnode = currentnode.next
            
            if currentnode.data == data:
                newNode.next = currentnode.next
                currentnode.next = newNode
                self.length += 1
                return
       
        print('Value does not exist')
        
    # method to delete node at the beginning
    def deleteBeg(self):
        if self.length == 0:
            print ('the list is empty')
        else:
            self.head = self.head.next
            self.length -= 1
            
    def deleteLast(self):
        if self.length == 0:
            print('the list is empty')
        else:
            currentnode = self.head
            previousnode = self.head
            while (currentnode.next != None):
               previousnode = currentnode
               currentnode = currentnode.next
               
        previousnode.next = None
        self.length -= 1
      
    # method to delete a node at a particular position
    def deleteAtPos(self, pos):
        count = 0
        currentnode = self.head
        previousnode = self.head
        
        if pos > self.length or pos < 0:
            print('The position is invalid')
            
        elif pos == 1:
            self.deleteBeg()
            self.length -= 1
        else:
            while(currentnode.next != None or count < pos):
                count = count+1
                if (count == pos):
                    previousnode.next = currentnode.next
                    self.length -= 1
                    return
                else:
                    previousnode = currentnode
                    currentnode = currentnode.next
                
    
    def deleteValue(self, data):
        currentnode = self.head
        previousnode = self.head
        
        while(currentnode.next != None or currentnode.data != data):
            previousnode = currentnode
            currentnode = currentnode.next
            
            if currentnode.data == data:
                previousnode.next = currentnode.next
                self.length -= 1
                return
            
        print('Value does not exist')
        
        
    def print_list(self):
        nodeList = []
        currentnode = self.head
        while currentnode != None:
            nodeList.append(currentnode.data)
            currentnode = currentnode.next
            
        print (nodeList)
    
    def getLength(self):
        return self.length

# Testing Linked List
        
if __name__ == "__main__":
    

    ll = LinkedList()
    
    # Note: I can not add or delete the same node twice
    
    nodes = [1,2,3,4,5,6]
    
    for i in nodes:
        node = Node(i)
        ll.addNode(node)
    
    print(ll.print_list())
    print(ll.getLength())
    
    # Add elements
    node1 = Node(9)
    node2 = Node(10)
    node3 = Node(11)
    node4 = Node(9)
    node5= Node(20)
    
    ll.addBeg(node1)
    print(ll.print_list())      
    
    ll.addLast(node2)
    print(ll.print_list())   
    
    ll.addAtPos(1, node3)
    print(ll.print_list())
    
    ll.addAtPos(2, node4)
    print(ll.print_list())
    
    ll.addAftervalue(3, node5)
    print(ll.print_list())
    
    # Delete elements
    ll.deleteBeg()
    print(ll.print_list())
    
    ll.deleteLast()
    print(ll.print_list())                
    
    ll.deleteAtPos(1) 
    print(ll.print_list())
    
    ll.deleteAtPos(2)
    print(ll.print_list())
    
    ll.deleteValue(20)
    print(ll.print_list())
    
    print(ll.getLength())