import os
import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from generate_landscapes import generate_fitness_landscape
from fitness_plots import generate_fitness_landscape_map, generate_fitness_heatmap, generate_peak_frequencies

if __name__ == "__main__":
    # Create 'figures' folder if it doesn't exist
    folder_name = 'figures'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
            
    # Number of genes
    N = 4

    # Degree of interaction
    K = 3

    # Alphabet to use
    alphabet = ['A', 'B', 'C', 'D']

    # Number of landscapes to generate
    num_landscapes = 30

    # Initialize seed once before generating landscapes
    np.random.seed()

    landscapes = []
    for i in range(num_landscapes):
        # Generate fitness landscape for current iteration
        fitness_landscape = generate_fitness_landscape(N, K, alphabet)

        # Append the generated landscape to the list
        landscapes.append(fitness_landscape)

        # Compare and rank all sequences by fitness
        sorted_fitness = sorted(fitness_landscape.items(), key=lambda x: x[1], reverse=True)

        # Check fitness value range and distribution
        fitness_values = list(fitness_landscape.values())
        print(f"Landscape {i + 1}: Fitness Value Range: {min(fitness_values)} - {max(fitness_values)}")

        # Convert sequences to a heatmap matrix
        sequences = list(fitness_landscape.keys())

        # Create a matrix for heatmap visualization
        sequence_matrix = np.array([[ord(char) - ord('A') for char in seq] for seq in sequences])
        fitness_matrix = np.array(fitness_values).reshape(len(alphabet)**N, 1)
        
        # Create a matrix for heatmap visualization
        sequence_matrix = np.array([[ord(char) - ord('A') for char in seq] for seq in sequences])
        fitness_matrix = np.array(fitness_values).reshape(len(alphabet)**N, 1)
        
        # Perform K-Means clustering
        combined_matrix = np.hstack((sequence_matrix, fitness_matrix))
        pca = PCA(n_components=2)
        reduced_combinations = pca.fit_transform(combined_matrix)
        kmeans = KMeans(n_clusters=10, random_state=0).fit(reduced_combinations)
        cluster_labels = kmeans.labels_

        # Calculate distances from cluster centers
        distances_from_center = np.sqrt(np.sum((reduced_combinations - kmeans.cluster_centers_[kmeans.labels_])**2, axis=1))
        
        # Define paths for saving plots
        landscape_folder = os.path.join(folder_name, f'landscape_{i + 1}')
        os.makedirs(landscape_folder, exist_ok=True)
        
        # Fit sequence values without stacking
        pca = PCA(n_components=2)
        reduced_combinations = pca.fit_transform(sequence_matrix)
                
        # Plotting the fitness landscape in 3D
        generate_fitness_landscape_map(reduced_combinations, fitness_values, save_path=os.path.join(landscape_folder, 'fitness_landscape_map.png'))

        # Plotting fitness heatmap
        generate_fitness_heatmap(sequence_matrix, fitness_matrix, sequences, save_path=os.path.join(landscape_folder, 'heatmap_with_sequences.png'))
        
        # Save sequences, clusters, distances, and fitness values to CSV
        filename = os.path.join(landscape_folder, f'landscape_{i + 1}_clusters.csv')
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Index', 'Sequence', 'Fitness', 'Cluster', 'Distance'])
            for seq_num, (seq, fitness, cluster, dist) in enumerate(zip(sequences, fitness_values, cluster_labels, distances_from_center)):
                writer.writerow([seq_num + 1, seq, fitness, cluster, dist])
        
    # Plotting frequencies of peaks in all fitness landscapes and isolating peaks
    all_peaks, all_peak_clusters = generate_peak_frequencies(landscapes, alphabet, save_path=os.path.join(folder_name, 'peak_frequencies_graph.png'))