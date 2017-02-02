"""
this file is a script to be run on monch to submit an array of jobs
"""
import sys
import os
version = str(sys.argv[1])
#version = 'v2'


model_sample_list = ['mock_12.0_1.0', 'mock_12.0_1.5', 'mock_12.0_2.0', 'mock_12.0_2.5', 'mock_12.0_3.0',
                        'mock_12.0_10.0', 'mock_12.5_1.0', 'mock_12.5_1.5'
                        ,'mock_12.5_2.0', 'mock_12.5_2.5', 'mock_12.5_3.0', 'mock_12.5_10.0', 'mock_13.0_1.0',
                        'mock_13.0_1.5', 'mock_13.0_2.0', 'mock_13.0_2.5', 'mock_13.0_3.0', 'mock_13.0_10.0',
                        'mock_13.5_1.0', 'mock_13.5_1.5', 'mock_13.5_2.0', 'mock_13.5_2.5', 'mock_13.5_3.0',
                        'mock_13.5_10.0']



for model in model_sample_list:
    submitstring = "sbatch -J "+ version+model + " array_exclusive.sh "+model+"_"+version
    os.system(submitstring)