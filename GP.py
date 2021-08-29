import math
import inspect
from random import random, randint, seed
from copy import deepcopy
from statistics import mean
import csv

POP_SIZE = 100  # population size
MIN_DEPTH = 3  # minimal initial random tree depth
MAX_DEPTH = 5  # maximal initial random tree depth
PROB_MUTATION = 0.9  # per-node mutation probability
XO_RATE = 0.3  # crossover rate
TOURNAMENT_SIZE = 10  # size of tournament for tournament selection
GENERATIONS = 1000  # maximal number of generations to run evolution

# seed(123456)


def add(x: float, y: float) -> float: return x + y
def sub(x: float, y: float) -> float: return x - y
def mul(x: float, y: float) -> float: return x * y
def div(x: float, y: float) -> float: return x/y if y != 0.0 else 1.0
def sin(x: float) -> float: return math.sin(x)
def cos(x: float) -> float: return math.cos(x)
def if_else_b(cond: bool, on_true: bool, on_false: bool) -> bool: return on_true if cond else on_false
def if_else_f(cond: bool, on_true: float, on_false: float) -> float: return on_true if cond else on_false
def is_less_than(x: float, y: float) -> bool: return x < y
def is_less_than_or_equal(x: float, y: float) -> bool: return x <= y
def is_grater_than(x: float, y: float) -> bool: return x > y
def is_grater_than_or_equal(x: float, y: float) -> bool: return x >= y
def or_b(x: bool, y: bool) -> bool: return x or y
def and_b(x: bool, y: bool) -> bool: return x or y
def exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return 10**10
def power(x: float, p: float) -> float:
    if x != 0:
        try:
            return (x**p).real
        except OverflowError:
            return 10**10
    else:
        return 0


FUNCTIONS = [add, sub, mul, div,
             sin, cos, power, exp,
             if_else_b, if_else_f, or_b, and_b,
             is_less_than, is_less_than_or_equal, is_grater_than, is_grater_than_or_equal]

VARIABLES_TUPLE = [('x', float)]
VARIABLES = []
TERMINALS = [-2.0, -1.0, 0.0, 1.0, 2.0, math.pi, math.e, True, False]
FUNCTIONS_RETURNS_TYPE = []
FUNCTIONS_PARAMS_TYPE = []
TERMINALS_TYPE = []


def set_terminals_type():
    for terminal in TERMINALS:
        TERMINALS_TYPE.append(type(terminal))
    for variable in VARIABLES_TUPLE:
        TERMINALS.append(variable[0])
        TERMINALS_TYPE.append(variable[1])


def get_parameter_types(function):
    parameter_types = []
    annotations = inspect.getfullargspec(function).annotations
    args_names = list(annotations.keys())
    for parameter in args_names:
        parameter_types.append(annotations.get(parameter))
    return parameter_types


def init_functs():
    for func in FUNCTIONS:
        types = get_parameter_types(func)
        FUNCTIONS_RETURNS_TYPE.append(types[0])
        types.reverse()
        types.pop()
        types.reverse()
        FUNCTIONS_PARAMS_TYPE.append(types)


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


init_functs()
set_terminals_type()


class Tree:
    def __init__(self, parent=None, children=None):
        self.parent = parent
        self.children = children

    def node_label(self):
        if self.parent in FUNCTIONS:
            return self.parent.__name__
        else:
            return str(self.parent)

    def print(self):
        print(self.tostring())

    def tostring(self):
        if self.parent in FUNCTIONS:
            parameters = ""
            if self.children:
                for child in self.children:
                    parameters += ", " + child.tostring()
                parameters = parameters[2:len(parameters)]
            return self.node_label() + "(" + parameters + ")"
        else:
            return self.node_label()

    def eval(self, x=0):
        if self.parent in FUNCTIONS:
            children_result = []
            if self.children:
                for child in self.children:
                    res = child.eval(x)
                    children_result.append(res)
                return self.parent(*children_result)
        if self.parent == 'x':
            return x
        else:
            return self.parent

    def add_random_function(self, parent, arg_index, pref_type=None):
        # FUNCTIONS = [ ..., function, ... ] <- parent
        # FUNCTIONS_PARAMS_TYPE = [ ..., [ ..., type_2, ... ], ... ] <- parent
        # FUNCTIONS_RETURN_Type = [ ..., function, ... ] <- random_function
        if parent:
            func_index = FUNCTIONS.index(parent)
            arg_type = FUNCTIONS_PARAMS_TYPE[func_index][arg_index]
            # selected random function most have the same return type as arg_type
            if arg_type is any:
                self.parent = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]
            else:
                allowed_funcs_index = indexes_of_in(arg_type, FUNCTIONS_RETURNS_TYPE)
                self.parent = FUNCTIONS[allowed_funcs_index[randint(0, len(allowed_funcs_index) - 1)]]
        else:
            if pref_type is not None:
                allowed_funcs_index = indexes_of_in(pref_type, FUNCTIONS_RETURNS_TYPE)
                self.parent = FUNCTIONS[allowed_funcs_index[randint(0, len(allowed_funcs_index) - 1)]]
            else:
                # in case of root
                self.parent = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]

    def add_random_terminal(self, parent, arg_index):
        func_index = FUNCTIONS.index(parent)
        arg_type = FUNCTIONS_PARAMS_TYPE[func_index][arg_index]
        # selected random terminal most have the same type as arg_type
        if arg_type is any:
            self.parent = TERMINALS[randint(0, len(TERMINALS) - 1)]
        else:
            allowed_terminal_index = indexes_of_in(arg_type, TERMINALS_TYPE)
            self.parent = TERMINALS[allowed_terminal_index[randint(0, len(allowed_terminal_index) - 1)]]

    def random_tree(self, grow, max_depth, depth=0, parent=None, arg_index=0, pref_type=None):
        if depth < MIN_DEPTH or (depth < max_depth and not grow):
            self.add_random_function(parent, arg_index, pref_type)
        elif depth >= max_depth:
            self.add_random_terminal(parent, arg_index)
        else:  # intermediate depth, grow
            if random() > 0.5:
                self.add_random_terminal(parent, arg_index)
            else:
                self.add_random_function(parent, arg_index, pref_type)

        if self.parent in FUNCTIONS:
            parameter_types = get_parameter_types(self.parent)
            parameters_number = len(parameter_types) - 1
            self.children = []
            for i in range(parameters_number):
                tree = Tree()
                tree.random_tree(grow, max_depth, depth=depth + 1, parent=self.parent, arg_index=i, pref_type=pref_type)
                self.children.append(tree)

    def mutation(self):
        if random() < PROB_MUTATION:
            if random() >= 0.5:
                if self.parent in FUNCTIONS:
                    func_index = indexes_of_in(self.parent, FUNCTIONS)
                    func_ret_type = FUNCTIONS_RETURNS_TYPE[func_index[0]]
                    self.random_tree(grow=True, max_depth=2, pref_type=func_ret_type)
            else:
                if self.children:
                    arg_index = randint(0, len(self.children)-1)
                    self.children[arg_index].mutation()
                else:
                    self.mutation()

    def type(self):
        if self.parent in FUNCTIONS:
            return FUNCTIONS_RETURNS_TYPE[indexes_of_in(self.parent, FUNCTIONS)[0]]
        else:
            return TERMINALS_TYPE[indexes_of_in(self.parent, TERMINALS)[0]]

    def crossover(self, other):
        if random() < XO_RATE:
            second = other.scan_tree([randint(2, other.size()+1)])
            positions = []
            self.search_for_matching_types(positions, second.type())
            if len(positions) > 1:
                cross_point = positions[randint(0, len(positions)-1)]
                self.cross(second, cross_point)
            elif len(positions) == 1:
                cross_point = positions[0]
                self.cross(second, cross_point)

    def scan_tree(self, count):
        count[0] -= 1
        if count[0] <= 1:
            return self.build_subtree()
        else:
            ret = None
            if self.children:
                for c in range(len(self.children)):
                    if self.children[c] and count[0] > 1:
                        ret = self.children[c].scan_tree(count)
            return ret

    def search_for_matching_types(self, positions, sub_type, id_list=[0]):
        if self.parent in TERMINALS:
            _id = id_list[-1]+1
            id_list.append(_id)
            if self.type() == sub_type:
                positions.append(_id)
            return positions
        else:
            _id = id_list[-1]+1
            id_list.append(_id)
            if self.type() == sub_type:
                positions.append(_id)
            if self.children:
                for child in self.children:
                    child.search_for_matching_types(positions, sub_type, id_list)

    def cross(self, sub_tree, position, id_list=[0]):
        if self.parent in TERMINALS:
            _id = id_list[-1] + 1
            id_list.append(_id)
            if position == _id:
                self.parent = sub_tree.parent
                if sub_tree.children:
                    self.children = sub_tree.children.copy()
        else:
            _id = id_list[-1] + 1
            id_list.append(_id)
            if position == _id:
                self.parent = sub_tree.parent
                if sub_tree.children:
                    self.children = sub_tree.children.copy()
            if self.children:
                for child in self.children:
                    child.cross(sub_tree, position, id_list=id_list)

    def build_subtree(self):
        t = Tree()
        t.parent = self.parent
        if self.children:
            t.children = self.children.copy()
        else:
            t.children = None
        return t

    def size(self):
        if self.parent in TERMINALS:
            return 1
        else:
            if self.children:
                s = 1
                for child in self.children:
                    s += child.size()
                return s
            else:
                return 1


def init_population(output_type: type):  # ramped half-and-half
    pop = []
    for i in range(int(POP_SIZE / 2)):
        t = Tree()
        t.random_tree(grow=True, max_depth=randint(3, MAX_DEPTH), pref_type=output_type)  # grow
        pop.append(t)
    for i in range(int(POP_SIZE / 2)):
        t = Tree()
        t.random_tree(grow=False, max_depth=randint(3, MAX_DEPTH), pref_type=output_type)  # full
        pop.append(t)
    return pop


def selection(population, fitnesses):  # select one individual using tournament selection
    tournament = [randint(0, len(population) - 1) for i in range(TOURNAMENT_SIZE)]
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]])


def fitness(individual, dataset):  # inverse mean absolute error over dataset normalized to [0,1]
    return 1 / (1 + mean([abs(individual.eval(ds[0]) - ds[1]) for ds in dataset]))


def target_func(x):
    return exp(sin(2*x)) + power(math.e, x/2)


def generate_dataset():  # generate 101 data points from target_func
    dataset = []
    for x in range(-100, 101, 2):
        x /= 10
        dataset.append([x, target_func(x)])
    return dataset


def load_database(path):
    data = []
    with open('data.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            data.append([float(row[0]), float(row[1])])
    return data


def main():
    dataset = generate_dataset()
    # dataset = load_database("data.csv")
    population = init_population(float)
    best_of_run = None
    best_of_run_f = 0
    best_of_run_gen = 0
    fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]

    # go evolution!
    for gen in range(GENERATIONS):
        nextgen_population = []
        for i in range(POP_SIZE):
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            parent1.crossover(parent2)
            parent1.mutation()
            nextgen_population.append(parent1)
        population = nextgen_population
        fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]

        best_of_run_f = max(fitnesses)
        best_of_run_gen = gen
        best_of_run = deepcopy(population[fitnesses.index(max(fitnesses))])
        print("________________________")
        print("Gen no.:", gen, ", Best:", round(max(fitnesses), 3), ", Mean:", round(mean(fitnesses), 3), ", Best sol.:")
        best_of_run.print()
        if best_of_run_f == 1:
            print("Done")
            break

    print("\n\n_________________________________________________\n"
          "END OF RUN\nBest attained at gen " + str(best_of_run_gen) +
          " and has fitness of " + str(round(best_of_run_f, 3)) + ".")
    best_of_run.print()


if __name__ == "__main__":
    main()

