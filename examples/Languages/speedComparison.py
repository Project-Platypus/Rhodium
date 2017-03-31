import os
import time
import subprocess

devnull = open(os.devnull, "w")
cwd = os.getcwd()
examples = ["Python", "C", "R", "Excel"]

for example in examples:
    os.chdir(os.path.join(cwd, example))
    start_time = time.time()
    subprocess.call(["python", "lakeModelIn" + example + ".py"], stdout=devnull)
    print("Lake Model in " + example + ": " + str(time.time() - start_time))
