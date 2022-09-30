from GP import evolve
from random import seed, randint

if __name__ == "__main__":
    seed_id = randint(0, 10**5)
    seed(seed_id)
    print("Run seed: ", seed_id)
    
    evolve()