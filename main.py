import Problem as Problem

import multiprocessing as mp

from Tree import Tree
from random import randint, seed
from copy import deepcopy
from statistics import mean

INDIVIDUAL_TABLE = []


def init_population(output_type: type):
    pop = []
    # full
    for i in range(int(Problem.POP_SIZE / 2)):
        t = Tree()
        t.random_tree(full=True, max_depth=randint(0, Problem.MAX_DEPTH), pref_type=output_type)
        pop.append(t)
    # grow
    for i in range(int(Problem.POP_SIZE / 2)):
        t = Tree()
        t.random_tree(full=False, max_depth=randint(0, Problem.MAX_DEPTH), pref_type=output_type)
        pop.append(t)
    return pop


def selection(population, fitness_values):  
    tournament = [randint(0, len(population) - 1) for _ in range(Problem.TOURNAMENT_SIZE)]
    tournament_fitness_values = [fitness_values[tournament[i]] for i in range(Problem.TOURNAMENT_SIZE)]
    return deepcopy(population[tournament[tournament_fitness_values.index(max(tournament_fitness_values))]])


def fitness(individual):
    st = str(individual)
    fv = None
    for record in INDIVIDUAL_TABLE:
        if record[0] == st:
            fv = record[1]
            break

    if fv is None:
        fv = Problem.evaluate_individual(individual, append_pid=Problem.DO_PARALLEL)
        INDIVIDUAL_TABLE.append((st, fv))

    return fv


def evolve():
    seed_id = randint(0, 10**5)
    seed(seed_id)
    print("Run seed: ", seed_id)

    # Parallel comutation
    if Problem.DO_PARALLEL:
        cores = mp.cpu_count()
        pool = mp.Pool(processes=cores)
        print(f"Parallel execution on {cores} cores is enabled.")
    else:
        print("Single core execution is enabled.")

    best_of_run = None
    best_of_run_f = 0
    best_of_run_gen = 0
    best_of_run_str = ""
    bests = []
    means = []
    best_individuals_str = []
    population = []

    # Evolution
    for gen in range(Problem.GENERATIONS):
        print("\n_________________")
        print(f"Generation no. {gen}")

        nextgen_population = []

        if gen == 0:
            # initiate the popuation 
            population = init_population(Problem.ROOT_TYPE)
            
        # evaluate the population
        fitness_values =\
            pool.map(fitness, population) if Problem.DO_PARALLEL\
            else [fitness(population[i]) for i in range(Problem.POP_SIZE)]

        fitness_values_p = fitness_values.copy()
        fitness_values_p.sort(reverse= not Problem.HIGHER_IS_BETTER)

        i = 0
        elites = []
        while len(elites) < Problem.NUMBER_OF_BESTS_TO_KEEP and i < Problem.POP_SIZE:
            elite_fitness = fitness_values_p[len(fitness_values_p)-1-i]
            elite_inedx = fitness_values.index(elite_fitness)
            elite = deepcopy(population[elite_inedx])
            if elite not in elites:
                elites.append(elite)
                nextgen_population.append(elite)
                print(f"\nEli {len(elites)} (Ind {elite_inedx}):\t[{round(elite_fitness, 2)}]\t\t{str(elite)}")
                elite.draw()
            i += 1
        print()

        for i in range(Problem.POP_SIZE-len(elites)):
            parent1 = selection(population, fitness_values)
            parent2 = selection(population, fitness_values)
            parent1.crossover(parent2)
            parent1.mutation()
            parent1.permutation()
            nextgen_population.append(parent1)
        
        # Print individuals
        if not Problem.DO_PARALLEL:
            order = [x for _, x in sorted(zip(fitness_values, range(0, Problem.POP_SIZE)), reverse=Problem.HIGHER_IS_BETTER)]
            for i in order: 
                print(f"Ind {i}:\t[{round(fitness_values[i], 2)}]\t\t{str(population[i])}")

        best_of_run_f = max(fitness_values)
        mean_of_run_f = mean(fitness_values)
        means.append(mean_of_run_f)
        bests.append(best_of_run_f)
        best_of_run_gen = gen
        best_of_run = elites[0]
        best_of_run_str = str(best_of_run)
        best_individuals_str.append(best_of_run_str)

        # Summary
        print()
        print(f"Mean fitness:\t[{round(mean_of_run_f, 2)}]")
        print(f"Best fitness:\t[{round(best_of_run_f, 2)}]")

        # Recording
        Problem.record(bests, means, best_individuals_str)

        # refill the population
        population = nextgen_population


    print("\n_________________________________________________\n"
          "Best attained at gen " + str(best_of_run_gen) +
          " and has fitness of " + str(round(best_of_run_f, 2)) + ".")
    print("Sol.:")
    print(best_of_run_str)
    best_of_run.draw()
    print()

if __name__ == "__main__":
    Problem.before_start()
    evolve()
    Problem.before_end()
    pass
