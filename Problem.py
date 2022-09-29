import math
import inspect
import os
import csv
from random import randint
from typing import List

# Problem definition
# ----------------------------------------------------------------------------------
HIGHER_IS_BETTER = True
DO_PARALLEL = False      # Execute individual evaluation's process in parallel 

POP_SIZE = 100            # population size
GENERATIONS = 15          # maximal number of generations to run evolution
TOURNAMENT_SIZE = POP_SIZE//10     # size of tournament for tournament selection
NUMBER_OF_BESTS_TO_KEEP = POP_SIZE//20

MAX_DEPTH = 6            # maximal initial random tree depth

PROB_MUTATION = 0.3      # per-node mutation probability
PROB_CROSSOVER = 0.8     # crossover rate
PROB_PERMUTATION = 0.3   # permutation on terminals

# Root type selection
ROOT_TYPE = List[float]

# -------------------------------------------------------------------------

# Funcrions defenition
def ADD(x1: float, x2: float) -> float: return x1+x2
def MUL(x1: float, x2: float) -> float: return x1*x2
def SUB(x1: float, x2: float) -> float: return x1-x2
def DIV(x1: float, x2: float) -> float: return x1/x2 if x2 != 0 else 0

def SQRT(x: float) -> float: 
    try: return math.sqrt(x)
    except: return 0
def POW(x1: float, x2: float) -> float: 
    try: return math.pow(x1, x2) 
    except: return 0
    

def SIN(x: float) -> float: return math.sin(x)
def COS(x: float) -> float: return math.cos(x)


# Function set defenition 
FUNCTIONS = [
    ADD, MUL, SUB, DIV, 
    POW, SQRT, 
    SIN, COS,
    ]

# Variable set defenition 
VARIABLES_TUPLE = [
    ('x', float),
    (-1.0, float), 
    (0.0, float), 
    (1.0, float), 
    (2.0, float), 
    (math.pi, float), 
    (math.e, float), 
]

VARIABLES = []
TERMINALS = []
FUNCTIONS_RETURNS_TYPE = []
FUNCTIONS_PARAMS_TYPE = []
TERMINALS_TYPE = []

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

def evaluate_individual(individual=None, str=None, summary=False, append_pid=False):
    return eval(individual, 1)


def record(bests, means, individuals):
    with open('temp/Fitness.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Best", "Mean", "Individual"])
        for i in range(len(bests)):
            writer.writerow([i, bests[i], means[i], str(individuals[i])])


def before_start():
    print("Running before start...")
    set_functions_type()
    set_terminals_type()


def before_end():
    print("Running before end...")
    pass



# -----------------------------------------------------------------------------------------
# Do not try to modify the following code.

def set_terminals_type():
    for terminal in TERMINALS:
        TERMINALS_TYPE.append(type(terminal))
    for variable in VARIABLES_TUPLE:
        TERMINALS.append(variable[0])
        TERMINALS_TYPE.append(variable[1])
        VARIABLES.append(variable[0])

def set_functions_type():
    for func in FUNCTIONS:
        types = get_return_and_parameters_types(func)
        FUNCTIONS_RETURNS_TYPE.append(types[0])
        types.reverse()
        types.pop()
        types.reverse()
        FUNCTIONS_PARAMS_TYPE.append(types)

def get_return_and_parameters_types(function):
    parameter_types = []
    annotations = inspect.getfullargspec(function).annotations
    args_names = list(annotations.keys())
    for parameter in args_names:
        parameter_types.append(annotations.get(parameter))

    return parameter_types

def indexes_of_in(item, space: list):
    start = 0
    length = len(space)
    positions = []
    while start <= length:
        try:
            index = space.index(item, start)
            if type(space[index]) == type(item):
                positions.append(index)
            start = index + 1

        except ValueError:
            break

    return positions if len(positions) > 0 else None

