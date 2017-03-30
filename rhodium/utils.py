from __future__ import print_function

import sys

def promptToRun(message, default="yes"):
    if default == "yes":
        prompt = "[Y/n]"
    elif default == "no":
        prompt = "[y/N]"
    else:
        raise ValueError("invalid default answer")
    
    print(message + " " + prompt + " ", end='')
    sys.stdout.flush()
    
    response = sys.stdin.readline().strip()
    
    if response == "":
        response = default[0]
        
    if response == "y" or response == "Y":
        return True
    else:
        return False