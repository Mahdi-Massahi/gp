IS_MAXIMIZATION = True             # Is this a maximization problem? 

DO_PARALLEL = True                 # Execute individual evaluation's process in parallel 

POP_SIZE = 100                     # population size
GENERATIONS = 15                   # maximal number of generations to run evolution
TOURNAMENT_SIZE = POP_SIZE//10     # size of tournament for tournament selection
ELITES_SIZE = POP_SIZE//20

MAX_DEPTH = 6                      # maximum tree depth

PROB_MUTATION = 0.3                # per-node mutation probability
PROB_CROSSOVER = 0.8               # crossover rate
PROB_PERMUTATION = 0.3             # permutation on terminals