# Implement PLA foir linearly separable dataset 
# Takes a csv of input data and location of output
# Run code as python3 problem1 input1.csv output1.csv 
# Weights in last line of output csv define decision boundary for given dataset 
# Weights are the slope of the decision boundary 

from visualize import *
import pandas as pd 
from visualize import visualize_scatter 
import numpy as np 
import sys 
import csv 

# takes in data, labels 
# P -> inputs with label 1
# N -> inputs with label -1
    # pick x in P or N
    # if x exists in P and w.x < 0:
         # w = w + x 
    # elif 
        # w = w - x
# alternatively: if w * x + b > 0, f(x) = 1 
# takes in data, label, weight, bias 

"""def format_output(output, output_file): 
    with open(output_file, mode = 'w') as file: 
        file_write = csv.writer(file, delimeter='')
        file_write.writerow(output)
"""

def PLT(d, l, w1, b, output_file):
    output = []
    conv = False  
    while not conv:
        w0 = w1 
        for x in range(d.shape[0]):
            if l[x] * (np.dot(w1, d[x]) +b) <= 0: 
                b = b + l[x]
                w1 = w1 + l[x] * d[x] 
                result = w1[0], w1[1]
        row = []
        row.extend(result)
        row.append(b)
        output.append(row)
        
        if np.all(w1 == w0):
            break  

        with open(output_file, 'w') as f:
            write = csv.writer(f, delimiter = ',')
            write.writerows(output)
    #print(output)

if __name__ == "__main__":

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    data = pd.read_csv('input1.csv', header=None)
    data = data.as_matrix()
    
    labels = data[:, 2]
    feature_data = data[:, 0:2]
    feat1 = data[:, 0]
    feat2 = data[:, 1]

    weight = np.zeros(feature_data.shape[1])
    bias = 0 

    PLT(feature_data, labels, weight, bias, output_file)



