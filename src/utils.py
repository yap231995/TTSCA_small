'''
Each of these functions will load a dataset and return the following datasets:
* The profiling traces and profiling labels (these will be split into training and validation)
* The attack traces and labels for each possible key guess
* The actual value of the target key byte
'''
import csv
import os
import numpy as np
import argparse



def str2list(x):
    x = str(x).replace(" ", "")
    if "[" in x:
        inputs_type2 = x.replace("[","").replace("]","")
        inputs_type1 = inputs_type2.split(",")
        z = [int(i) for i in inputs_type1]
    return z

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def printListofList(lst):
    for i in range(len(lst)):
        print(i)
        for ele2 in lst[i]:
            print(ele2)

def Save_CSV_Lst(lst, str_name, Path):
    path = os.path.join(Path, str_name +".csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(lst)

def Load_CSV_Lst(str_name, Path):
    path = os.path.join(Path, str_name + '.csv')
    with open(path, "r") as f:
        reader = csv.reader(f)
        lst = list(reader)
    return lst


def get_reference1_2_3(num_of_variables = 17):
    """
    Input:
    num_of_variables: total number of variables x_1 to x_num and ~x_1 to ~x_num
    Output:
    reference1: dictionary (key,values): number in base 2*num_of_variables+1, x_i
    reference2: dictionary (key,values): number in base 2*num_of_variables+1, (i,1 if x_i but -1 if ~x_i)
    reference3: dictionary (key,values): x_i, number in base 2*num_of_variables+1

    """
    reference1 = {}
    reference2 = {}
    reference3 = {}
    for kkkk in range(1,num_of_variables+1):
        reference1[str(np.base_repr(kkkk,2*num_of_variables+1))] = "x_"+str(kkkk-1)
        reference2[str(np.base_repr(kkkk,2*num_of_variables+1))] = (kkkk-1, 1)
        reference3["x_" + str(kkkk - 1)] = np.base_repr(kkkk, 2*num_of_variables+1)
        # if kkkk == num_of_variables:
        #     reference1[str(np.base_repr(kkkk,2*num_of_variables+1))] = "y"
        #     reference2[str(np.base_repr(kkkk,2*num_of_variables+1))] = (-1, 1)
        #     reference3["y"] = np.base_repr(kkkk, 2*num_of_variables+1)
    for kkkk in range(num_of_variables+1,2*num_of_variables+1):
        reference1[str(np.base_repr(kkkk,2*num_of_variables+1))] = "~x_" + str(kkkk-num_of_variables+1)
        reference2[str(np.base_repr(kkkk,2*num_of_variables+1))] = (kkkk-num_of_variables+1, -1)
        reference3["~x_" + str(kkkk - num_of_variables+1)] = np.base_repr(kkkk, 2*num_of_variables+1)
        # if kkkk == 2*num_of_variables:
        #     reference1[str(np.base_repr(kkkk,2*num_of_variables+1))] = "~y"
        #     reference2[str(np.base_repr(kkkk,2*num_of_variables+1))] = (-1, -1)
        #     reference3["~y"] = np.base_repr(kkkk, 2*num_of_variables+1)
    # print("reference1: ",reference1) ## dictionary (key,values): number in base 2*num_of_variables+1, x_i
    # print("reference2: ",reference2) ## dictionary (key,values): number in base 2*num_of_variables+1, (i,1 if x_i but -1 if ~x_i)
    # print("reference3: ",reference3) ## dictionary (key,values): x_i, number in base 2*num_of_variables+1
    return reference1, reference2, reference3