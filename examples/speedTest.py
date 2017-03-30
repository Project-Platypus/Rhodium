import os
import timeit

cwd = os.getcwd()

os.chdir(os.path.join(cwd, "C"))
print("Lake Model in C")
print(timeit.timeit("exec(open('lakeModelInC.py').read())", number=1))

print()
    
os.chdir(os.path.join(cwd, "R"))
print("Lake Model in R")
print(timeit.timeit("exec(open('lakeModelInR.py').read())", number=1))