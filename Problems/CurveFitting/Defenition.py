from math import pi, e, sqrt, pow, sin, cos, exp
from turtle import color

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
def MUL(x1: float, x2: float) -> float: return x1*x2
def SUB(x1: float, x2: float) -> float: return x1-x2
def DIV(x1: float, x2: float) -> float: return x1/x2 if x2 != 0 else 0
def SQRT(x: float) -> float: 
    try: return sqrt(x)
    except: return 0
def POW(x1: float, x2: float) -> float: 
    try: return pow(x1, x2) 
    except: return 0
def SIN(x: float) -> float: return sin(x)
def COS(x: float) -> float: return cos(x)
def EXP(x: float) -> float: return exp(x)


# FUNCTIONS holds the function set of the GP problem
# Note:
# 1. define functions in FunctionSet.py 
# 2. Include those which are needed here in this list 
# Note: make sure to provide a sufficent set of functions
FUNCTIONS = [
    ADD, MUL, SUB, DIV, 
    # POW, SQRT, 
    SIN, COS,
    EXP,
    ]


# TERMINAL_TUPLES holds touples of the terminals 
# the first element holds the value or varibale name,
# the second element holds the type of the first elemnt
# Note: 
# Make sure to provide a sufficent set of functions
TERMINAL_TUPLES = [
    # Variables
    ('x', float),

    # Literals 
    (-1.0, float), 
    (0.0, float), 
    (1.0, float), 
    (2.0, float), 
    (pi, float), 
    (e, float), 
]


# ROOT_TYPE defines the type of the solution's ourput which GP provides
# In other words, it is the type of the root of the trees which GP provides
ROOT_TYPE = float


# cerate custum target dataset by a custum formula
def target_func(x):
    # return x*sin(x)
    return sin(x)/x if x != 0 else 0
    # return -123.4
    # return sin(x) + x + 2
    # return x * log(x)

def generate_dataset():
    dataset = []
    for x in list(np.arange(-10, 10, 0.5)):
        dataset.append([x if x != 0 else 0, target_func(x)])
    return dataset

TARGET_DATASET = generate_dataset()


# Evaluation function 
def eval(individual, x):
    if individual.parent in FUNCTIONS:
        children_result = []
        if individual.children:
            for child in individual.children:
                res = eval(child, x)
                children_result.append(res)
            return individual.parent(*children_result)
    if individual.parent == 'x':
        return x
    else:
        return individual.parent


# Returns the fitness of the individuals
def evaluate_individual(individual=None, append_pid=False):
    fitness = None
    try: 
        fitness = sqrt(sum([pow(eval(individual, tds[0]) - tds[1], 2) for tds in TARGET_DATASET]))
    except:
        fitness = -(10**5) if IS_MAXIMIZATION else (10**5)
    return fitness


# This function is called before the GP run
def internal_before_start():
    global FIG, AXS
    FIG, AXS = plt.subplots(1, 2, figsize=(10, 5))


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
    FIG.suptitle(f"Gen no.: {len(bests)}, Fitness:{round(best_of_run, 2)}", fontsize=10)
    AXS[0].plot(np.array(bests), color='r', linewidth=1, label="Best")
    AXS[0].plot(np.array(means), color='b', linewidth=1, label="Mean")

    # plot the traget and the best curve
    y = []
    x = []
    target = []
    for ds in TARGET_DATASET:
        x.append(ds[0])
        y.append(eval(individuals[-1], ds[0]))
        target.append(ds[1])
    AXS[1].cla()
    AXS[1].plot(x, target, label="Target", color='r', linewidth=2)
    AXS[1].plot(x, y, label="Best", color='b', linewidth=0.5)

    # pause the plot
    plt.pause(0.001)


# This function is called after the GP run
def internal_before_end():
    plt.show()