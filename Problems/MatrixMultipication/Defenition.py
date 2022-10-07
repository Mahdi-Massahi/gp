from cmath import sqrt
from statistics import mean, stdev
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import csv

from .Hyperparameters import IS_MAXIMIZATION


# In this script, one should define the functions which are 
# used as a function in the function set of the GP problem
# Note: 
# 1. Functions must be explecitly typed; 
#    Types of the arguments and return value must be explicitly defined.
# 2. Functions must be fault tolerant;
#    make sure the defined function covers the desired domain.
#    return a "safty value" in case of evaluation failure 
# 3. Feel free to use any name as function arguments
def ADD(x1: float, x2: float) -> float: return x1+x2
def SUB(x1: float, x2: float) -> float: return x1-x2
def MUL(x1: float, x2: float) -> float: return x1*x2

def ADD_AT_BY(
        x1: float,
        row_x2: int, col_x2: int, mat2: List[float]) -> float: 
    return x1+mat2[row_x2][col_x2]

def SUB_AT_BY(
        x1: float,
        row_x2: int, col_x2: int, mat2: List[float]) -> float: 
    return x1-mat2[row_x2][col_x2]

def MUL_AT_BY(
        x1: float,
        row_x2: int, col_x2: int, mat2: List[float]) -> float: 
    return x1*mat2[row_x2][col_x2]

def ADD_AT(
        row_x1: int, col_x1: int, mat1: List[float],
        row_x2: int, col_x2: int, mat2: List[float]) -> float: 
    return mat1[row_x1][col_x1]+mat2[row_x2][col_x2]

def SUB_AT(
        row_x1: int, col_x1: int, mat1: List[float],
        row_x2: int, col_x2: int, mat2: List[float]) -> float: 
    return mat1[row_x1][col_x1]-mat2[row_x2][col_x2]

def MUL_AT(
        row_x1: int, col_x1: int, mat1: List[float],
        row_x2: int, col_x2: int, mat2: List[float]) -> float: 
    return mat1[row_x1][col_x1]*mat2[row_x2][col_x2]

class ROOT: pass
def ASIGN(pos1: float, pos2: float, pos3: float, pos4: float) -> ROOT: 
    return [[pos1, pos2], [pos3, pos4]]



# FUNCTIONS holds the function set of the GP problem
# Note:
# 1. define functions in FunctionSet.py 
# 2. Include those which are needed here in this list 
# Note: make sure to provide a sufficent set of functions
FUNCTIONS = [
    # ADD, SUB, MUL,
    ADD_AT, MUL_AT, SUB_AT,
    ASIGN,
    ]

# TERMINAL_TUPLES holds touples of the terminals 
# the first element holds the value or varibale name,
# the second element holds the type of the first elemnt
# Note: 
# Make sure to provide a sufficent set of functions
TERMINAL_TUPLES = [
    # Variables
    ('mat1', List[float]),
    ('mat2', List[float]),

    # Literals 
    (0, int), 
    (1, int), 
]


# ROOT_TYPE defines the type of the solution's ourput which GP provides
# In other words, it is the type of the root of the trees which GP provides
ROOT_TYPE = ROOT


# cerate custum target dataset by a custum formula
def target_func(mat1, mat2):
    return list(np.matmul(np.array(mat1), np.array(mat2)))

def generate_dataset():
    dataset = []
    for _ in range(1):
        mat1 = np.random.rand(2, 2)
        mat2 = np.random.rand(2, 2)
        dataset.append([mat1, mat2, target_func(mat1, mat2)])
    return dataset

TARGET_DATASET = generate_dataset()



# Evaluation function 
def eval(individual, mat1, mat2) -> np.ndarray:
    if individual.parent in FUNCTIONS:
        children_result = []
        if individual.children:
            for child in individual.children:
                res = eval(child, mat1, mat2)
                children_result.append(res)
            return individual.parent(*children_result)
    if individual.parent == 'mat1':
        return mat1
    elif individual.parent == 'mat2':
        return mat2
    else:
        return individual.parent


def evaluate_individual(individual=None, append_pid=False):
    sses = []
    for test_case in TARGET_DATASET:
        residual = np.array(eval(individual, test_case[0], test_case[1])) - test_case[2]
        sses.append(abs(sqrt(np.sum(np.power(residual, 2)))))
    return sum(sses)


# This function is called before the GP run
def internal_before_start():
    fig1 = plt.figure(1)

# This function is called at the end of the each iteration
# It can be used to save records or plot
def internal_before_iteration(bests, means, individuals):
    # saving the fitness records on file
    with open('temp/Fitness.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Best", "Mean", "Individual"])
        for i in range(len(bests)):
            writer.writerow([i, bests[i], means[i], str(individuals[i])])

    # plotting the fitness values
    best_of_run = max(bests) if IS_MAXIMIZATION else min(bests)
    plt.title(f"Gen no.: {len(bests)}, Fitness:{round(best_of_run, 5)}", fontsize=10)
    plt.plot(np.array(bests), color='r', linewidth=1, label="Best")
    plt.plot(np.array(means), color='b', linewidth=1, label="Mean")
    plt.pause(0.001)

# This function is called after the GP run
def internal_before_end():
    plt.show()