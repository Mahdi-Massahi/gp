import math
import inspect
from random import random, randint, seed

MIN_DEPTH = 2  # minimal initial random tree depth
MAX_DEPTH = 15  # maximal initial random tree depth

# seed(123456)


def add(x: float, y: float) -> float: return x + y
def sub(x: float, y: float) -> float: return x - y
def mul(x: float, y: float) -> float: return x * y
def div(x: float, y: float) -> float: return x/y if y != 0 else 1
def sin(x: float) -> float: return math.sin(x)
def cos(x: float) -> float: return math.cos(x)
def if_else_b(cond: bool, on_true: bool, on_false: bool) -> bool: return on_true if cond else on_false
def if_else_f(cond: bool, on_true: float, on_false: float) -> float: return on_true if cond else on_false
def is_less_than(x: float, y: float) -> bool: return x < y
def is_less_than_or_equal(x: float, y: float) -> bool: return x <= y
def is_grater_than(x: float, y: float) -> bool: return x > y
def is_grater_than_or_equal(x: float, y: float) -> bool: return x >= y
def power(x: float, p: float) -> float: return (x**p).real if x != 0 else 0
def exp(x: float) -> float: return math.exp(x)
def or_b(x: bool, y: bool) -> bool: return x or y
def and_b(x: bool, y: bool) -> bool: return x or y


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
        print(self.__tostring())

    def __tostring(self):
        if self.parent in FUNCTIONS:
            parameters = ""
            if self.children:
                for child in self.children:
                    parameters += ", " + child.__tostring()
                parameters = parameters[2:len(parameters)]
            return self.node_label() + "(" + parameters + ")"
        else:
            return self.node_label()

    def eval(self, x=0):
        if self.parent in FUNCTIONS:
            children_result = []
            for child in self.children:
                children_result.append(child.eval(x))
            return self.parent(*children_result)
        if self.parent == 'x':
            return x
        else:
            return self.parent

    def add_random_function(self, parent, arg_index):
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

    def random_tree(self, grow, max_depth, depth=0, parent=None, arg_index=0):
        if depth < MIN_DEPTH or (depth < max_depth and not grow):
            self.add_random_function(parent, arg_index)
        elif depth >= max_depth:
            self.add_random_terminal(parent, arg_index)
        else:  # intermediate depth, grow
            if random() > 0.5:
                self.add_random_terminal(parent, arg_index)
            else:
                self.add_random_function(parent, arg_index)
        if self.parent in FUNCTIONS:
            parameter_types = get_parameter_types(self.parent)
            parameters_number = len(parameter_types) - 1
            self.children = []
            for i in range(parameters_number):
                tree = Tree()
                tree.random_tree(grow, max_depth, depth=depth + 1, parent=self.parent, arg_index=i)
                self.children.append(tree)


tree = Tree()
tree.random_tree(True, 15)

tree.print()
print(tree.eval(10))

