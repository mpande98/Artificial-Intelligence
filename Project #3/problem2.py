# Use gradient descent to build linear regression model for predicting 
# height, age, weight 
# output alpha, num_iters, bias, b_age, b_weight

import pandas as pd 
import sys 
import numpy as np 
import csv 
from pandas import DataFrame

learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

# add vector column for the intercept at the front of the matrix 

# get mean for each feature column 
def get_mean(x):
    #return sum(x)/len(x)
    return np.mean(x, axis = 0)
    
# get standard deviation for each feature column 
def get_sd(x):
    return np.std(x, axis =0)
    # return np.std(x, axis = 0)

# scale each feature (height and weight) by its standard deviation 
def scale_feature(f):
    return (f - get_mean(f))/get_sd(f)

# run this method with each alpha 100 times 
# pick a 10th learning rate 
def gradient_descent(num_iter, feat_data, Y, output_file): 
    # 3 dimensions, 1 intercept at the front, 2 features 
    beta = np.zeros(feat_data.shape[1]) 
    output = []
    for alpha in learning_rates: 
        count = 0 
        # f(xi) - yi
        for i in range(num_iter): 
            count +=1
            error = np.dot(feat_data, beta) - Y 
        #R = np.sum(error ** 2)
        # now we are calculaing gradient descent rule 
            for x in range(feat_data.shape[1]): 
                beta[x] -= (alpha * (1/len(Y))) * np.sum(error * feat_data[:, x])
            result = beta[0], beta[1], beta[2]
        row = [alpha]
        row.append(count)
        row.extend(result)
        output.append(row)
            
    with open(output_file, 'w') as f:
        write = csv.writer(f, delimiter = ',')
        write.writerows(output)
    #return alpha, num_iter, beta[0], beta[1], beta[2]

def gradient_descent_add(new_alpha, new_iter, feat_data, Y, output_file): 
    beta = np.zeros(feat_data.shape[1]) 
    count = 0
    output = []
    for i in range(new_iter): 
        count +=1
        error = np.dot(feat_data, beta) - Y 
        for x in range(feat_data.shape[1]): 
            beta[x] -= (new_alpha * (1/len(Y))) * np.sum(error * feat_data[:, x])
        result = beta[0], beta[1], beta[2]
    row = [new_alpha]
    row.append(count)
    row.extend(result)
    output.append(row)

    with open(output_file, 'a') as f:
        write = csv.writer(f, delimiter = ',')
        write.writerows(output)


# https://stackoverflow.com/questions/29287224/pandas-read-in-table-without-headers
if __name__ == "__main__":

    input_file = sys.argv[1]
    data = pd.read_csv('input2.csv', header=None)
    # type is numpy 
    data = data.as_matrix()
    
    # get number of rows or dimension 
    dim = data.shape[1]
    # add intercept at the beginning 
    data = np.insert(data, obj = 0, values =1, axis = 1) 
    
    # age and weight are features 
    intercept = data[:, 0]
    age = data[:, 1]
    weight = data[:, 2]
    height = data[:, 3]

    # this would be the middle 2 columns
    feature_data = data[:,1:3]
    #print(feature_data)
    output_csv = sys.argv[2]
    scaled_age = scale_feature(age)
    scaled_weight = scale_feature(weight)
    normalized_feature_data = np.column_stack((intercept, scaled_age, scaled_weight))
    #print(normalized_feature_data)
    num_iter = 100 
    #out_file = open(output_csv, 'w')
  
    """for alpha in learning_rates:
        i = 0 
        for each in range(num_iter):
            i = i + 1 
            result = gradient_descent(alpha, i, normalized_feature_data, height)
            print(result)
    """
    gradient_descent(num_iter, normalized_feature_data, height, output_csv)
    gradient_descent_add(0.8, 70, normalized_feature_data, height, output_csv)
    
    # To graph the data 
    #output_data = pd.read_csv('output2.csv', header= None)
    """alpha = output_data[:, 0]
    print(alpha)
    num_iter = output_data[:, 1]
    b_0 = output_data[:, 2]
    b_age = output_data[:, 3]
    b_weight = output_data[:, -1]
    lin_reg_weights = output_data[:,2:5]
    print(lin_reg_weights)
    """
    # plot each feature on xy plane 
    # plot regressionn equation as plane in xyz space 