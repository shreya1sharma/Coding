#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 09:29:12 2018

@author: shreya
"""

from linked_list import LinkedList, Node


#iterative solution
def iterReverse(ll):
    currentnode = ll.head
    nextnode = currentnode.next
    previousnode = None
        
    while(currentnode!= None):
        nextnode = currentnode.next
        currentnode.next = previousnode
        previousnode = currentnode
        currentnode = nextnode
          
    ll.head = previousnode
    

#recursive solution
def recReverse(ll, node):
    if node.next == None:
        ll.head = node
        return
    
    recReverse(ll, node.next)
    temp = node.next
    temp.next = node
    node.next = None
    


if __name__ == "__main__":
    ll = LinkedList()
    nodes = [1,2,3,4,5,6]

    for i in nodes:
        node = Node(i)
        ll.addNode(node)
        
    print(ll.print_list())
    
    iterReverse(ll)
    print(ll.print_list())
    
    recReverse(ll, ll.head)
    print(ll.print_list())