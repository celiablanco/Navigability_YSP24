# Original code adapted from: Maciej Workiewicz
# Source: https://github.com/Mac13kW/NK_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from typing import List

def create_interaction_matrix(N: int, K: int) -> np.ndarray:
    """
    Creates a random interaction matrix for NK landscape.

    Parameters:
    - N (int): Number of decision variables.
    - K (int): Number of interactions per variable.

    Returns:
    - np.ndarray: Random interaction matrix of shape (N, N).
    """
    interaction_matrix = np.zeros((N, N), dtype=int)
    for i in range(N):
        if K > 0:
            choices = np.random.choice(N, size=K, replace=False)
            interaction_matrix[i, choices] = 1
    return interaction_matrix

def generate_nk_landscape(N: int) -> np.ndarray:
    """
    Generates a random NK landscape.

    Parameters:
    - N (int): Number of decision variables.

    Returns:
    - np.ndarray: NK landscape matrix of shape (2^N, N).
    """
    num_combinations = 2 ** N
    return np.random.rand(num_combinations, N)

def generate_power_key(N: int) -> np.ndarray:
    """
    Generates a power key for converting binary combinations to indices.

    Parameters:
    - N (int): Number of decision variables.

    Returns:
    - np.ndarray: Power key array of shape (N,).
    """
    return 2 ** np.arange(N)[::-1]

def calculate_fitness(nk_landscape: np.ndarray, interaction_matrix: np.ndarray, current_position: np.ndarray, power_key: np.ndarray) -> np.ndarray:
    """
    Calculate fitness values for each decision variable based on NK landscape.

    Parameters:
    - nk_landscape (np.ndarray): NK landscape data of shape (2**N, N) where N is the number of decision variables.
    - interaction_matrix (np.ndarray): Interaction matrix of shape (N, N) defining interactions among decision variables.
    - current_position (np.ndarray): Current combination of decision variables, represented as a binary array of length N.
    - power_key (np.ndarray): Power key used to find addresses on the landscape, typically calculated as np.power(2, np.arange(N - 1, -1, -1)).

    Returns:
    - np.ndarray: Fitness vector of length N representing fitness values for each decision variable.
    """
    N = len(current_position)
    fitness_vector = np.zeros(N)
    for i in range(N):
        index = np.sum(current_position * interaction_matrix[i] * power_key)
        fitness_vector[i] = nk_landscape[int(index), i]
    return fitness_vector

def create_mount_fuji_landscape(N: int) -> np.ndarray:
    """
    Creates a Mount Fuji landscape with a single global peak.

    Parameters:
    - N (int): Number of decision variables.

    Returns:
    - np.ndarray: Mount Fuji landscape of shape (2^N,).
    """
    num_combinations = 2 ** N
    landscape = np.zeros(num_combinations)
    peak_index = num_combinations // 2
    for i in range(num_combinations):
        distance = np.sum(np.binary_repr(i, width=N) != np.binary_repr(peak_index, width=N))
        landscape[i] = np.exp(-distance)
    return landscape

def simulate_landscape(nk_landscape: np.ndarray, interaction_matrix: np.ndarray, power_key: np.ndarray) -> np.ndarray:
    """
    Simulates the landscape and calculates fitness values for all combinations.

    Parameters:
    - nk_landscape (np.ndarray): The NK landscape.
    - interaction_matrix (np.ndarray): Interaction matrix defining interactions between decision variables.
    - power_key (np.ndarray): Array used to convert binary combinations to indices.

    Returns:
    - np.ndarray: Array containing combinations, fitness vectors, average fitness, local peaks, and global peak.
    """
    N = interaction_matrix.shape[0]
    num_combinations = 2 ** N
    landscape_data = np.zeros((num_combinations, N*2+3))

    for c1, c2 in enumerate(product(range(2), repeat=N)):
        combination_array = np.array(c2)
        fit_vector = calculate_fitness(nk_landscape, interaction_matrix, combination_array, power_key)
        landscape_data[c1, :N] = combination_array
        landscape_data[c1, N:2*N] = fit_vector
        landscape_data[c1, 2*N] = np.mean(fit_vector)

    for c3 in range(num_combinations):
        is_local_peak = 1
        for c4 in range(N):
            neighbor = landscape_data[c3, :N].copy().astype(int)
            neighbor[c4] = 1 - neighbor[c4]
            neighbor_index = np.sum(neighbor * power_key)
            if landscape_data[c3, 2*N] < landscape_data[int(neighbor_index), 2*N]:
                is_local_peak = 0
        landscape_data[c3, 2*N+1] = is_local_peak

    global_peak_index = np.argmax(landscape_data[:, 2*N])
    landscape_data[global_peak_index, 2*N+2] = 1
    return landscape_data

def generate_multiple_nk_landscapes(N: int, num_landscapes: int) -> np.ndarray:
    """
    Generates multiple NK landscapes.

    Parameters:
    - N (int): Number of decision variables.
    - num_landscapes (int): Number of NK landscapes to generate.

    Returns:
    - np.ndarray: 3D array of NK landscapes of shape (num_landscapes, 2**N, N).
    """
    landscapes = np.zeros((num_landscapes, 2**N, N))
    for i in range(num_landscapes):
        landscapes[i] = generate_nk_landscape(N)
    return landscapes

def plot_number_of_peaks_histogram(number_of_peaks):
    """
    Plots the histogram of the number of peaks and saves it as an image file.

    Parameters:
    - number_of_peaks (np.ndarray): Array of the number of peaks in each landscape.

    Returns:
    - None: Saves the histogram plot as an image file.
    """
    plt.figure(1, facecolor='white', figsize=(8, 6), dpi=150)
    plt.hist(number_of_peaks, bins=20, range=(1, 20), color='dodgerblue', edgecolor='black')
    plt.title('Distribution of the Number of Peaks', size=12)
    plt.xlabel('Number of Peaks', size=10)
    plt.ylabel('Frequency', size=10)
    plt.savefig('number_of_peaks_histogram.png')
    plt.close()

def analyze_landscapes_for_N_K_values(N_values: List[int], K_values: List[int], num_landscapes_per_combination: int) -> None:
    """
    Analyzes NK landscapes for different combinations of N and K values.

    Parameters:
    - N_values (list): List of N values (number of decision variables).
    - K_values (list): List of K values (number of interactions per variable).
    - num_landscapes_per_combination (int): Number of landscapes to generate for each (N, K) combination.

    Returns:
    - None: Writes the summary table to a CSV file and plots the distribution of peaks.
    """
    results = []
    all_number_of_peaks = []
    sequence_fitness_records = []

    for N in N_values:
        for K in K_values:
            if K < N:
                number_of_peaks_list = []
                max_values_list = []
                min_values_list = []
                
                for _ in range(num_landscapes_per_combination):
                    if K == 0:
                        landscape_data = create_mount_fuji_landscape(N)
                        
                        # Mount Fuji landscape has one peak
                        number_of_peaks = 1
                        max_value = np.max(landscape_data)
                        min_value = np.min(landscape_data)
                        
                        number_of_peaks_list.append(number_of_peaks)
                        max_values_list.append(max_value)
                        min_values_list.append(min_value)
                        
                        for idx, fitness_value in enumerate(landscape_data):
                            binary_sequence = np.binary_repr(idx, width=N)
                            sequence_fitness_records.append([binary_sequence, fitness_value])
                    else:
                        interaction_matrix = create_interaction_matrix(N, K)
                        power_key = np.power(2, np.arange(N - 1, -1, -1))
                        nk_landscape = np.random.rand(2**N, N)
                        landscape_data = simulate_landscape(nk_landscape, interaction_matrix, power_key)
                        
                        number_of_peaks = np.sum(landscape_data[:, 2*N+1])
                        max_value = np.max(landscape_data[:, 2*N])
                        min_value = np.min(landscape_data[:, 2*N])
                        
                        number_of_peaks_list.append(number_of_peaks)
                        max_values_list.append(max_value)
                        min_values_list.append(min_value)
                        
                        for row in landscape_data:
                            binary_sequence = ''.join(map(str, row[:N].astype(int)))
                            avg_fitness = row[2*N]
                            sequence_fitness_records.append([binary_sequence, avg_fitness])

                all_number_of_peaks.extend(number_of_peaks_list)

                avg_number_of_peaks = np.mean(number_of_peaks_list)
                max_peak_value = np.max(max_values_list)
                min_peak_value = np.min(min_values_list)
                
                results.append([N, K, np.sum(number_of_peaks_list), avg_number_of_peaks, min_peak_value, max_peak_value])
                
    columns = ['N', 'K', 'Total_Peaks', 'Avg_Number_of_Peaks', 'Min_Peak_Value', 'Max_Peak_Value']
    df = pd.DataFrame(results, columns=columns)
    df.to_csv('nk_landscape_analysis.csv', index=False)

    plot_number_of_peaks_histogram(all_number_of_peaks)

    sequence_fitness_df = pd.DataFrame(sequence_fitness_records, columns=['Sequence', 'Fitness'])
    sequence_fitness_df.to_csv('sequence_fitness_values.csv', index=False)

if __name__ == "__main__":
    N_values = [5, 6, 7]  # Example values for N
    K_values = [0, 1, 2, 3, 4, 5]  # Example values for K
    num_landscapes_per_combination = 10  # Number of landscapes to generate per (N, K) combination

    analyze_landscapes_for_N_K_values(N_values, K_values, num_landscapes_per_combination)
