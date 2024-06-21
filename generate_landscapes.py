import numpy as np
from itertools import product

def generate_combinations(N, alphabet):
    return [''.join(comb) for comb in product(alphabet, repeat=N)]

def generate_fitness_contributions(N, K, alphabet):
    contributions = {}
    for i in range(N):
        contributions[i] = {}
        for comb in generate_combinations(K + 1, alphabet):
            contributions[i][comb] = np.random.rand()
    return contributions

def calculate_fitness(sequence, fitness_contributions, K):
    N = len(sequence)
    total_fitness = 0
    for i in range(N):
        # Adjust the window size to ensure the key length matches K+1
        start = max(0, i - K // 2)
        end = start + K + 1
        local_sequence = sequence[start:end]
        
        if local_sequence in fitness_contributions[i]:
            total_fitness += fitness_contributions[i][local_sequence]
        else:
            total_fitness += 0
    return total_fitness / N

def generate_fitness_landscape(N, K, alphabet):
    combinations = generate_combinations(N, alphabet)
    fitness_contributions = generate_fitness_contributions(N, K, alphabet)
    fitness_landscape = {comb: calculate_fitness(comb, fitness_contributions, K) for comb in combinations}
    return fitness_landscape