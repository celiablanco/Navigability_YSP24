import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.cluster.hierarchy import linkage, dendrogram
from collections import defaultdict

def generate_fitness_landscape_map(reduced_combinations, fitness_values, save_path=None):
    """Plot the fitness landscape in 3D using a surface plot."""
    # Create a figure for the 3D plot
    fig = plt.figure(figsize=(10, 7))
    
    # Add a 3D subplot
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract the PCA-reduced coordinates
    xs = reduced_combinations[:, 0]
    ys = reduced_combinations[:, 1]
    zs = fitness_values
    
    # Generate a grid for the surface plot
    xi = np.linspace(xs.min(), xs.max(), 100)
    yi = np.linspace(ys.min(), ys.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((xs, ys), zs, (xi, yi), method='cubic')
    
    # Surface plot with colors representing fitness values
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none')
    
    # Labeling the axes
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('Fitness')
    
    # Set the title of the plot
    ax.set_title('NK Fitness Landscape Visualization')
    
    # Add a color bar to show the mapping of fitness values to colors
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Fitness')

    # Save the surface plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
    
def generate_fitness_heatmap(sequence_matrix, fitness_matrix, sequences, save_path=None):
    # Perform hierarchical clustering
    linkage_matrix = linkage(sequence_matrix, method='average')
    dendro = dendrogram(linkage_matrix, no_plot=True)
    ordered_indices = dendro['leaves']

    # Reorder the fitness matrix based on the clustering
    ordered_fitness_matrix = fitness_matrix[ordered_indices, :]

    # Heatmap visualization with clustering
    plt.figure(figsize=(12, 8))
    sns.heatmap(ordered_fitness_matrix, annot=False, fmt=".4f", cmap="viridis", cbar_kws={'label': 'Fitness'})
    plt.title("NK Fitness Landscape Heatmap with Clustering")
    plt.xlabel("Sequence Number")
    plt.ylabel("Fitness Value")
    
    # Replace sequence indices with sequence numbers
    plt.yticks(ticks=range(len(ordered_indices)), labels=[str(idx + 1) for idx in ordered_indices])
    
    # Save the heatmap
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
    
def is_peak(seq, fitness_landscape, alphabet):
    """Check if a sequence is a peak (local maximum) in the fitness landscape."""
    current_fitness = fitness_landscape[seq]
    N = len(seq)
    for i in range(N):
        for letter in alphabet:
            if letter != seq[i]:
                neighbor = seq[:i] + letter + seq[i+1:]
                if fitness_landscape[neighbor] > current_fitness:
                    return False
    return True

def isolate_and_count_peaks(fitness_landscape, alphabet):
    """Isolate count peaks"""
    peaks = [seq for seq in fitness_landscape if is_peak(seq, fitness_landscape, alphabet)]
    peak_clusters = defaultdict(list)
    
    for peak in peaks:
        peak_clusters[peak].append(peak)
        for seq in fitness_landscape:
            if seq != peak and fitness_landscape[seq] < fitness_landscape[peak]:
                peak_clusters[peak].append(seq)

    return peaks, peak_clusters

def generate_peak_frequencies(landscapes, alphabet, save_path=None):
    """Plot frequencies of peaks in all landscapes."""
    all_peaks = []
    all_peak_clusters = defaultdict(list)
    
    for fitness_landscape in landscapes:
        peaks, peak_clusters = isolate_and_count_peaks(fitness_landscape, alphabet)
        all_peaks.extend(peaks)
        
        for peak, cluster in peak_clusters.items():
            all_peak_clusters[peak].extend(cluster)
    
    # Count the frequency of peaks
    peak_counts = defaultdict(int)
    for peak in all_peaks:
        peak_counts[peak] += 1

    # Create a frequency distribution of the number of peaks
    peak_frequencies = defaultdict(int)
    for peak in all_peaks:
        peak_frequencies[peak_counts[peak]] += 1

    # Prepare data for plotting
    x = list(peak_frequencies.keys())
    y = list(peak_frequencies.values())

    # Plot the number of peaks vs. frequency
    plt.figure(figsize=(10, 7))
    plt.bar(x, y, color='skyblue')
    plt.xlabel('Number of Peaks')
    plt.ylabel('Frequency')
    plt.title('Number of Peaks vs. Frequency in NK Fitness Landscape')
    plt.xticks(x)

    # Save the frequency plot if a path is provided
    if save_path:
        plt.savefig(save_path)
    plt.close()

    return all_peaks, all_peak_clusters