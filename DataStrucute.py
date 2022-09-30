import inspect
from typing import List
from random import random, randint
from copy import deepcopy

from Problems.config import *

VARIABLES = []
TERMINALS = []
FUNCTIONS_RETURNS_TYPE = []
FUNCTIONS_PARAMS_TYPE = []
TERMINALS_TYPE = []

class Tree:
    def __init__(self, parent=None, children=None):
        self.parent = parent
        self.children = children
        self.id = -1
        self.depth_index = -1

    def __eq__(self, other):
        str_1 = self.tostring()
        str_2 = other.tostring()
        return str_1 == str_2

    def __str__(self) -> str:
        if self.parent in FUNCTIONS:
            parameters = ""
            if self.children:
                for child in self.children:
                    parameters += ", " + str(child)
                parameters = parameters[2:len(parameters)]
            return self.node_label() + "(" + parameters + ")"
        else:
            return self.node_label()

    def tostring(self) -> str:
        return str(self)

    def node_label(self) -> str:
        if self.parent in FUNCTIONS:
            return self.parent.__name__
        else:
            if type(self.parent) == float:
                return str(round(self.parent, 2))
            else:
                return str(self.parent)

    def draw(self, prefix=" ", is_corner=False, debug=False):
        if debug and prefix == " ":
            self.update_node_ids()
            self.update_depth_indexes()
        # Do not try to underestand the following code!
        label = self.node_label() + ((" (i" + str(self.id) + ', d' + str(self.depth_index) + ")") if debug else "") 
        if prefix[len(prefix)-1] == '─': pre = prefix
        else:
            index_before_space = [i for i, c in enumerate(prefix) if c == "│"]
            if len(index_before_space) > 0:
                if is_corner: pre = prefix[0:index_before_space[-1]] + "├" + ("─"*(len(prefix)-1-index_before_space[-1]))
                else: pre = prefix[0:index_before_space[-1]] + "└" + ("─"*(len(prefix)-1-index_before_space[-1]-1))
            else: pre = prefix
        prefix += "│" 
        pre = pre.replace("│.", " ")
        ex =  "○ " if not self.children else "⦿ "
        print(pre + ex + label, sep="", end="\n")
        if self.children:
            children_number = len(self.children)
            for child_i in range(children_number):
                is_corner = False
                if child_i == children_number-1: prefix = prefix + '.'
                else: is_corner = True
                self.children[child_i].draw(prefix + " "*(len(label)), is_corner=is_corner, debug=debug)

    def size(self) -> int:
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

    def depth(self, depth=0) -> int:
        # the root has depth of 1 
        if self.children is None:
            return depth + 1
        if self.children:
            child_depth = []
            for child in self.children:
                child_depth.append(child.depth()+1)
            return max(child_depth) 
        
    def type(self) -> type:
        if self.parent in FUNCTIONS:
            return FUNCTIONS_RETURNS_TYPE[self.indexes_of_in(self.parent, FUNCTIONS)[0]]
        elif self.parent in TERMINALS:
            return TERMINALS_TYPE[self.indexes_of_in(self.parent, TERMINALS)[0]]
        else:
            return type(self.parent)

    def add_random_function(self, parent, arg_index, pref_type=None):
        # FUNCTIONS = [ ..., function, ... ] <- parent
        # FUNCTIONS_PARAMS_TYPE = [ ..., [ ..., type_2, ... ], ... ] <- parent
        # FUNCTIONS_RETURN_Type = [ ..., function, ... ] <- random_function
        if parent:
            func_index = FUNCTIONS.index(parent)
            arg_type = FUNCTIONS_PARAMS_TYPE[func_index][arg_index]
            # selected random function most have the same return type as arg_type
            allowed_funcs_index = self.indexes_of_in(arg_type, FUNCTIONS_RETURNS_TYPE)
            if allowed_funcs_index:
                self.parent = FUNCTIONS[allowed_funcs_index[randint(0, len(allowed_funcs_index) - 1)]]
            else:
                self.add_random_terminal(parent, arg_index)

        else:
            if pref_type is not None:
                allowed_funcs_index = self.indexes_of_in(pref_type, FUNCTIONS_RETURNS_TYPE)
                if allowed_funcs_index:
                    self.parent = FUNCTIONS[allowed_funcs_index[randint(0, len(allowed_funcs_index) - 1)]]
                else:
                    self.add_random_terminal(parent, arg_index, pref_type)
            else:
                # in case of root
                self.parent = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]

    def add_random_terminal(self, parent, arg_index=0, pref_type=None):
        if pref_type is None:
            func_index = FUNCTIONS.index(parent)
            pref_type = FUNCTIONS_PARAMS_TYPE[func_index][arg_index]
        allowed_terminal_index = self.indexes_of_in(pref_type, TERMINALS_TYPE)
        if allowed_terminal_index:
            self.parent = TERMINALS[allowed_terminal_index[randint(0, len(allowed_terminal_index) - 1)]]
        else:
            self.add_random_function(parent, arg_index)

    def generate_random_terminal(self):
        allowed_terminal_index = self.indexes_of_in(self.type(), TERMINALS_TYPE)
        if allowed_terminal_index:
            self.parent = TERMINALS[allowed_terminal_index[randint(0, len(allowed_terminal_index) - 1)]]

    def random_tree(self, full, max_depth, depth=1, parent=None, arg_index=0, pref_type=None):
        if depth == 1:
            self.add_random_function(parent, arg_index, pref_type)
        elif depth < max_depth:
            if not full:
                if random() > 0.5:
                    self.add_random_function(parent, arg_index, pref_type)
                else:
                    self.add_random_terminal(parent, arg_index)
            else:
                self.add_random_function(parent, arg_index, pref_type)
        else:
            self.add_random_terminal(parent, arg_index)

        # if parent is choosed to be a function, fill the children too,
        if self.parent in FUNCTIONS:
            return_and_parameter_types = self.get_return_and_parameters_types(self.parent)
            parameters_number = len(return_and_parameter_types) - 1
            self.children = []
            for i in range(parameters_number):
                tree = Tree()
                tree.random_tree(
                    full=full,
                    max_depth=max_depth, 
                    depth=depth+1,
                    parent=self.parent,
                    arg_index=i,
                    pref_type=None)
                self.children.append(tree)

    def get_terminal_ids(self, ids, __id=None):
        if __id is None:
            __id = [0]
        if self.parent in TERMINALS:
            ids.append(__id[0])
            return
        if self.children:
            for child in self.children:
                __id[0] += 1
                child.get_terminal_ids(ids, __id)

    def update_depth_indexes(self, depth=0):
        # the root has depth of 1 
        self.depth_index = depth + 1
        if self.children:
            for child in self.children:
                child.update_depth_indexes(depth+1)

    def update_node_ids(self, __id=None):
        # id is indexed starting at 0. Root's id is 0.
        if __id is None:
            __id = [0]        
        self.id = __id[0]
        if self.children:
            for child in self.children:
                __id[0] += 1
                child.update_node_ids(__id)
    
    def search_node_id(self, node_id: int, __is_node_found: List[bool]=None, action="", args=[]):
        if __is_node_found is None:
            __is_node_found = [False]
            self.update_node_ids()
            self.update_depth_indexes()

        if self.id == node_id:
            # target node is found
            if action == "mutate":
                self.mutate()
            if action == "cross":
                self.cross(*args)
            if action == "permutate":
                self.permutate()
            return
        else:
            if self.children:
                for child in self.children:
                    child.search_node_id(node_id, __is_node_found, action, args)
                    if __is_node_found[0] == True: return

    def mutation(self):
        if random() < PROB_MUTATION:
            mutation_point = randint(0, self.size()-1)
            self.search_node_id(node_id=mutation_point, action="mutate", args=[])

    def mutate(self):
        max_allowed_depth = MAX_DEPTH-self.depth_index
        if max_allowed_depth > 0:
            self.random_tree(full=True if random() > 0.5 else False,
                            max_depth=randint(0, max_allowed_depth),
                            pref_type=self.type())

    def crossover(self, other):
        if random() < PROB_CROSSOVER:
            crossover_point = randint(0, self.size()-1)
            self.search_node_id(node_id=crossover_point, action="cross", args=[other])

    def cross(self, other):
        subtree_type = self.type()
        subtrees = []
        other.get_type_matching_subtrees(subtree_type, subtrees)
        if len(subtrees) > 0:
            max_allowed_depth = MAX_DEPTH - self.depth_index 
            allowed_subtrees = [subtree for subtree in subtrees if subtree.depth() <= max_allowed_depth]
            if len(allowed_subtrees) > 0:
                subtree_other = deepcopy(allowed_subtrees[randint(0, len(allowed_subtrees)-1)])
                self.parent = subtree_other.parent
                self.children = subtree_other.children

    def permutation(self):
        if random() < PROB_PERMUTATION:
            ids = []
            self.get_terminal_ids(ids)
            permutation_point = ids[randint(0, len(ids)-1)]
            self.search_node_id(node_id=permutation_point, action="permutate", args=[])
    
    def permutate(self):
        self.add_random_terminal(self, pref_type=self.type())

    def get_type_matching_subtrees(self, subtree_type, subtrees=None):
        if subtrees is None:
            subtrees = []
        if self.type() == subtree_type:
            subtrees.append(self)
        if self.children:
            for child in self.children:
                child.get_type_matching_subtrees(subtree_type, subtrees)

    @staticmethod
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

    @staticmethod
    def get_return_and_parameters_types(function):
        parameter_types = []
        annotations = inspect.getfullargspec(function).annotations
        args_names = list(annotations.keys())
        for parameter in args_names:
            parameter_types.append(annotations.get(parameter))

        return parameter_types

    @staticmethod
    def set_terminals_type():
        for terminal in TERMINALS:
            TERMINALS_TYPE.append(type(terminal))
        for variable in TERMINAL_TUPLES:
            TERMINALS.append(variable[0])
            TERMINALS_TYPE.append(variable[1])
            VARIABLES.append(variable[0])

    @staticmethod
    def set_functions_type():
        for func in FUNCTIONS:
            types = Tree.get_return_and_parameters_types(func)
            FUNCTIONS_RETURNS_TYPE.append(types[0])
            types.reverse()
            types.pop()
            types.reverse()
            FUNCTIONS_PARAMS_TYPE.append(types)

