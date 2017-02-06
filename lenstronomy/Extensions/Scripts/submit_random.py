"""
this file is a script to be run on monch to submit an array of jobs
"""
import sys
import os
version = str(sys.argv[1])
num = int(sys.argv[2])
#version = 'v2'

name = 'mock_'

for i in range(num):
    job_name = name+str(i)+"_"+version
    submitstring = "sbatch -J "+ version+str(i) + " array_exclusive.sh "+job_name
    os.system(submitstring)