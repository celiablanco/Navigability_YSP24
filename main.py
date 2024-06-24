import os
import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from generate_landscapes import generate_fitness_landscape
from fitness_plots import generate_fitness_landscape_map, generate_fitness_heatmap, generate_peak_frequencies

def summarize_landscapes(landscapes, alphabet, N, K, save_path):
    summary_data = []
    for i, fitness_landscape in enumerate(landscapes):
        fitness_values = list(fitness_landscape.values())
        num_peaks = len(generate_peak_frequencies([fitness_landscape], alphabet, save_path=None)[0])
        max_fitness = max(fitness_values)
        min_fitness = min(fitness_values)
        summary_data.append([i + 1, N, K, num_peaks, max_fitness, min_fitness])
    
    # Write summary to CSV
    summary_file = os.path.join(save_path, f'landscape_summary_N{N}_K{K}.csv')
    with open(summary_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Landscape Index', 'N', 'K', 'Number of Peaks', 'Max Fitness', 'Min Fitness'])
        writer.writerows(summary_data)

if __name__ == "__main__":
    # Base folder for all figures
    base_folder = 'figures'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    # Number of landscapes to generate
    num_landscapes = 30

    # Initialize seed once before generating landscapes
    np.random.seed()

    # Varying N and K values
    varying_parameters = [(4, 3), (5, 2), (6, 1)]

    # Alphabet
    alphabet = ["A", "B", "C", "D"]

    for N, K in varying_parameters:
        # Create a folder for each parameter configuration
        config_folder = os.path.join(base_folder, f'N{N}_K{K}')
        if not os.path.exists(config_folder):
            os.makedirs(config_folder)
        
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
            print(f"Landscape {i + 1} for N={N}, K={K}: Fitness Value Range: {min(fitness_values)} - {max(fitness_values)}")

            # Convert sequences to a heatmap matrix
            sequences = list(fitness_landscape.keys())
            fitness_values = list(fitness_landscape.values())

            # Create a matrix for heatmap visualization
            sequence_matrix = np.array([[ord(char) - ord('A') for char in seq] for seq in sequences])
            fitness_matrix = np.array(fitness_values).reshape(len(alphabet)**N, 1)

            # Perform K-Means clustering with combined matrix (sequence + fitness values)
            combined_matrix = np.hstack((sequence_matrix, fitness_matrix))
            pca_combined = PCA(n_components=2)
            reduced_combined = pca_combined.fit_transform(combined_matrix)
            kmeans_combined = KMeans(n_clusters=10, random_state=0).fit(reduced_combined)
            cluster_labels_combined = kmeans_combined.labels_

            # Calculate distances from cluster centers (combined)
            distances_from_center_combined = np.sqrt(np.sum((reduced_combined - kmeans_combined.cluster_centers_[kmeans_combined.labels_])**2, axis=1))
            
            # Define paths for saving plots within the configuration folder
            landscape_folder = os.path.join(config_folder, f'landscape_{i + 1}')
            os.makedirs(landscape_folder, exist_ok=True)
            
            # Perform PCA and K-Means clustering without fitness values (only sequences)
            pca_sequence = PCA(n_components=2)
            reduced_sequence = pca_sequence.fit_transform(sequence_matrix)
            kmeans_sequence = KMeans(n_clusters=10, random_state=0).fit(reduced_sequence)
            cluster_labels_sequence = kmeans_sequence.labels_

            # Plotting the fitness landscape in 3D based on sequence values
            generate_fitness_landscape_map(reduced_sequence, fitness_values, save_path=os.path.join(landscape_folder, 'fitness_landscape_map.png'))

            # Plotting fitness heatmap
            generate_fitness_heatmap(sequence_matrix, fitness_matrix, sequences, save_path=os.path.join(landscape_folder, 'heatmap_with_sequences.png'))
            
            # Save sequences, clusters, distances, and fitness values to CSV
            filename = os.path.join(landscape_folder, f'landscape_{i + 1}_clusters.csv')
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Index', 'Sequence', 'Fitness', 'Cluster (Combined)', 'Distance (Combined)', 'Cluster (Sequence)', 'Distance (Sequence)'])
                for seq_num, (seq, fitness, cluster_comb, dist_comb, cluster_seq, dist_seq) in enumerate(zip(
                    sequences, fitness_values, cluster_labels_combined, distances_from_center_combined, cluster_labels_sequence, np.sqrt(np.sum((reduced_sequence - kmeans_sequence.cluster_centers_[kmeans_sequence.labels_])**2, axis=1)))):
                    writer.writerow([seq_num + 1, seq, fitness, cluster_comb, dist_comb, cluster_seq, dist_seq])
        
        # Summarize landscapes for the current N and K values
        summarize_landscapes(landscapes, alphabet, N, K, config_folder)
    
    # Summarize all landscapes together in the base folder
    all_landscapes = []
    for N, K in varying_parameters:
        for i in range(num_landscapes):
            fitness_landscape = generate_fitness_landscape(N, K, alphabet)
            all_landscapes.append(fitness_landscape)

    # Plotting frequencies of peaks in all fitness landscapes and isolating peaks
    all_peaks, all_peak_clusters = generate_peak_frequencies(all_landscapes, alphabet, save_path=os.path.join(base_folder, 'peak_frequencies_graph.png'))