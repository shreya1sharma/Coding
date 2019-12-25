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
#%% iterables, generators and yield
#iterables: can be iterated over and over again eg. lists, strings, etc
mylist=[1,2,3]
for i in mylist:
    print(i)
mylist=[x*x for x in range(3)]
for i in mylist:
    print(i)
    
#generators: a kind of iterable you can only iterate over once, do not store all the values in memory and they generate values on the fly
mygenerator= (x*x for x in range(3))
for i in mygenerator:
    print(i)
#will not iterate second time when I call it
for i in mygenerator:
    print(i)

#yield: a keyword that is used like 'return', except the function will return a generator
def createGenerator():
    mylist= range(3)
    for i in mylist:
        yield i*i

mygenerator = createGenerator()
print(mygenerator) # does not return i*i but returns only generator object
for i in mygenerator:
    print(i)
#%%enumerate
example= ['left', 'right','up','down']
for i in range(len(example)):
    print(i, example[i])
    
for i,j in enumerate(example): #enumerate function returns a tuple containing the index and the actual value from the iterable, the iterable can be lists, dictionary ets.
    print(i,j)
    
example_dict= {'left':'<', 'right':'>', 'up':'^','down':'v',}
[print(i,j) for i,j in enumerate(example_dict)]

new_dict= dict(enumerate(example))
print(new_dict)

print(list(enumerate(example)))
#%% zip function: iterates through multiple iterables and aggregates them
x= [1,2,3,4]
y= [7,8,3,2]
z= ['a','b','c','d']
#2 values
for a,b in zip(x,y):
    print(a,b)
#3 values
for a,b,c in zip(x,y,z):   
    print(a,b,c)
#zip object
print(zip(x,y,z))
print(list(zip(x,y,z)))   

#creating a dictionary
names= ['Jill', 'Jack','Jeb','Jessica']
grades= [99,56,24,87]

d= dict(zip(names, grades))
print(d)   

#list comprehension v/s for loop
x=[1,2,3,4]
y=[7,8,3,2]
z=['a','b','c','d']

[print(x,y,z) for x,y,z in zip(x,y,z)] 
print(x)   #x is still [1,2,3,4]

for x,y,z in zip(x,y,z):
    print(x,y,z)
print(x)  #new x overwrites the original x in regular for loop
#%% Creating generator functions   
def simple_gen():
    yield 'Oh'
    yield 'hello'

for i in simple_gen():
    print(i)

#example
#apprach 1
correct_combo=(3,6,1)    
for c1 in range(10):
    for c2 in range(10):
        for c3 in range(10):
            if(c1,c2,c3)==correct_combo:
                print('found the combo:{}'.format((c1,c2,c3)))

#apprach 2
for c1 in range(10):
    for c2 in range(10):
        for c3 in range(10):
            if(c1,c2,c3)==correct_combo:
                print('found the combo:{}'.format((c1,c2,c3)))
                break                
  
#approach 3: breaking loic for each line
found_combo= False
for c1 in range(10):
    if found_combo:
        break
    for c2 in range(10):
        if found_combo:
            break
        for c3 in range(10):
            if(c1,c2,c3)==correct_combo:
                print('found the combo:{}'.format((c1,c2,c3)))
                found_combo=True
                break                
  
#approach 4: More pythonic
def combo_gen():
    for c1 in range(10):
        for c2 in range(10):
            for c3 in range(10):
                yield (c1,c2,c3)

for (c1,c2,c3) in combo_gen():
    print(c1,c2,c3)       
    if (c1,c2,c3)== correct_combo:
        print('Found the combo :{}'.format((c1,c2,c3)))
        break
#%% multiprocessing
import multiprocessing
def spawn(num1, num2):
    print('Spawned # {} {}'.format(num1, num2))

if __name__=='__main__':
    for i in range(5):
        p= multiprocessing.Process(target= spawn, args=(i,i+1))
        p.start()
        #p.join() #this maintains the order but does not utilise multiprocessing