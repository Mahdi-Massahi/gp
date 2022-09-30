from math import pi, e, sqrt, pow, sin, cos

import csv


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



# FUNCTIONS holds the function set of the GP problem
# Note:
# 1. define functions in FunctionSet.py 
# 2. Include those which are needed here in this list 
# Note: make sure to provide a sufficent set of functions
FUNCTIONS = [
    ADD, MUL, SUB, DIV, 
    POW, SQRT, 
    SIN, COS,
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


def evaluate_individual(individual=None, append_pid=False):
    return eval(individual, 1)


# This function is called before the GP run
def internal_before_start():
    pass

# This function is called at the end of the each iteration
# It can be used to save records or plot
def internal_before_iteration(bests, means, individuals):
    with open('temp/Fitness.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Best", "Mean", "Individual"])
        for i in range(len(bests)):
            writer.writerow([i, bests[i], means[i], str(individuals[i])])

# This function is called after the GP run
def internal_before_end():
    pass