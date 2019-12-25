# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:25:58 2018

@author: hoge
"""
#argparse module is used to write user-friendly CLI
import argparse
import sys

def main():
    parser= argparse.ArgumentParser()
    parser.add_argument('--x', type= float, default= 1.0, help= 'what is the first number?')
    parser.add_argument('--y', type= float, default= 1.0, help= 'what is the second number?')
    parser.add_argument('--operation', type= str, default='add', help='what operation? Can schoose add, sub, mul or div')
    args= parser.parse_args()   #args in an object which stores the variable and the operations
    sys.stdout.write(str(calc(args)))
    
def calc(args):
    if args.operation =='add':
        return args.x+args.y
    elif args.operation=='sub':
        return args.x-args.y
    elif args.operation=='mul':
        return args.x*args.y
    elif args.operation=='div':
        return args.x/args.y
    
if __name__=='__main__':
    main()
    
'''
#save the py script in C:/Users/hoge
#on the anaconda cmd prompt type: python arg_parse_example -h
#to add python and conda in windows cmd:
    1. open anaconda cmd
    2. get the location of python.exe and conda.exe using( where conda, where python)
    3. add these path in system enviroment variables using system properites like this:
       3.1 Computer -> System Properties (or Win+Break) -> Advanced System Settings
       3.2 Click the Environment variables... button (in the Advanced tab)
       3.3  Edit PATH and append  ";path of python.exe; path of conda.exe" to the end
       3.4 Click OK. Note that changes to the PATH are only reflected in command prompts opened after the change took place
       3.5 Type python in windows cmd to check if the path is saved
     Alternately, setx path command can also be used
#references:
    https://stackoverflow.com/questions/4621255/how-do-i-run-a-python-program-in-the-command-prompt-in-windows-7
    https://medium.com/@GalarnykMichael/install-python-on-windows-anaconda-c63c7c3d1444
'''  
        
