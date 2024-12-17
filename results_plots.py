import argparse
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import numpy as np
import os
from scipy.stats import linregress

def plot_metrics(data):
    """Given the dictionary of the results -> Plot the results"""
    # Create the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Plot training and validation loss
    axes[0].plot(data["Step"], data["Training Loss"], label="Training Loss", marker="o", color="blue")
    axes[0].plot(data["Step"], data["Validation Loss"], label="Validation Loss", marker="o", color="orange")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Plot validation accuracy
    axes[1].plot(data["Step"], data["Validation Accuracy"], label="Validation Accuracy", marker="o", color="green")
    axes[1].set_title("Validation Accuracy Curve")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    # Adjust layout and display
    plt.tight_layout()
    return plt.show()

def plot_lora_svd_singular_values(svd_results):
    """
    Plot singular values, with the x-axis as the index and y-axis as the value of the singular value.
    One row per layer, columns for matrix types.
    """
    num_layers = len(svd_results)
    matrix_types = ['query', 'key', 'value']
    
    # Create figure with one row per layer
    fig, axes = plt.subplots(num_layers, 3, figsize=(15, 4*num_layers))
    fig.suptitle('Singular Values in LoRA Weights', fontsize=16)
    
    for layer_idx, (layer_key, layer_data) in enumerate(svd_results.items()):
        for matrix_idx, matrix_name in enumerate(matrix_types):
            singular_values = layer_data[matrix_name]
            
            # Plot singular values
            axes[layer_idx, matrix_idx].plot(range(len(singular_values)), singular_values, marker='o')
            axes[layer_idx, matrix_idx].set_title(f'{layer_key} - {matrix_name} Matrix')
            axes[layer_idx, matrix_idx].set_xlabel('Index')
            axes[layer_idx, matrix_idx].set_ylabel('Singular Value')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()

def plot_log_lora_svd_singular_values(svd_results):
    """
    Plot singular values, with the x-axis as the log index and y-axis as the value of the singular value.
    One row per layer, columns for matrix types.
    """
    num_layers = len(svd_results)
    matrix_types = ['query', 'key', 'value']
    
    # Create figure with one row per layer
    fig, axes = plt.subplots(num_layers, 3, figsize=(15, 4*num_layers))
    fig.suptitle('Singular Values in LoRA Weights', fontsize=16)
    
    for layer_idx, (layer_key, layer_data) in enumerate(svd_results.items()):
        for matrix_idx, matrix_name in enumerate(matrix_types):
            singular_values = np.log(layer_data[matrix_name])
            
            # Plot singular values
            axes[layer_idx, matrix_idx].plot(range(len(singular_values)), singular_values, marker='o')
            axes[layer_idx, matrix_idx].set_title(f'{layer_key} - {matrix_name} Matrix')
            axes[layer_idx, matrix_idx].set_xlabel('Index')
            axes[layer_idx, matrix_idx].set_ylabel('Singular Value')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()


def plot_histogram(data, bins=10, title='Distribution Histogram', xlabel='Value', file_path=None):
    """
    Plot a histogram of floating-point values.
    """
    # Convert input to numpy array
    data_array = np.array(data)
    
    # Create figure and axis
    plt.figure(figsize=(8, 5))
    
    # Plot histogram
    plt.hist(data_array, bins=bins, edgecolor="black")
    
    # Set labels and title
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    
    # Add grid for better readability
    plt.grid(linestyle='--')
    
    # Compute and display some basic statistics
    plt.annotate(f'Mean: {np.mean(data_array):.2f}\n'
                 f'Median: {np.median(data_array):.2f}\n'
                 f'Std Dev: {np.std(data_array):.2f}', 
                 xy=(0.95, 0.95), xycoords='axes fraction', 
                 horizontalalignment='right', 
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # Adjust layout
    plt.tight_layout()

    if file_path:
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        plt.savefig(file_path)
    
    return plt.show()

def test_log_relationship(singular_values, layer_name, matrix_name):
    """
    Test if there is a logarithmic relationship between the index and singular values.
    Plots the original and log-transformed relationships and computes a linear fit.
    """
    indices = np.arange(1, len(singular_values) + 1)  # Avoid log(0) by starting from 1
    log_values = np.log(singular_values)
    
    # Fit a linear model in log space
    slope, intercept, r_value, p_value, std_err = linregress(indices, log_values)
    
    # Plot the original data
    plt.figure(figsize=(12, 6))
    
    # Original plot
    plt.subplot(1, 2, 1)
    plt.plot(indices, singular_values, marker='o', label='Original')
    plt.title(f'Original Singular Values ({layer_name} - {matrix_name})')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.grid(True)
    plt.legend()
    
    # Log-transformed plot
    plt.subplot(1, 2, 2)
    plt.plot(indices, log_values, marker='o', label='Log-Transformed')
    plt.plot(indices, intercept + slope * indices, label=f'Fit (r={r_value:.2f})', linestyle='--')
    plt.title(f'Log Relationship ({layer_name} - {matrix_name})')
    plt.xlabel('Index')
    plt.ylabel('Log(Singular Value)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Log Fit: Slope = {slope:.2f}, Intercept = {intercept:.2f}, R-squared = {r_value**2:.2f}")
    return r_value**2  # Coefficient of determination (goodness of fit)

def test_power_relationship(singular_values, layer_name, matrix_name):
    """
    Test if there is a power-law relationship between the index and singular values.
    Plots the log-log relationship and computes a linear fit.
    """
    indices = np.arange(1, len(singular_values) + 1)  # Avoid log(0) by starting from 1
    log_indices = np.log(indices)
    log_values = np.log(singular_values)
    
    # Fit a linear model in log-log space
    slope, intercept, r_value, p_value, std_err = linregress(log_indices, log_values)
    
    # Plot the log-log data
    plt.figure(figsize=(12, 6))
    
    # Log-log plot
    plt.plot(log_indices, log_values, marker='o', label='Log-Log Transformed')
    plt.plot(log_indices, intercept + slope * log_indices, label=f'Fit (r={r_value:.2f})', linestyle='--')
    plt.title(f'Power-Law Relationship ({layer_name} - {matrix_name})')
    plt.xlabel('Log(Index)')
    plt.ylabel('Log(Singular Value)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Power-Law Fit: Slope = {slope:.2f}, Intercept = {intercept:.2f}, R-squared = {r_value**2:.2f}")
    return r_value**2  # Coefficient of determination (goodness of fit)





import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Function 1: Plot intrinsic dimension coefficients over runs
def plot_coefficients_over_runs(
    results, 
    coefficient_name, 
    x_axis="num_classes", 
    layers=None, 
    matrix_types=None, 
    combine_runs=True, 
    filter_runs=None, 
    title=None
):
    """
    Plot intrinsic dimensionality coefficients (gini, elbow, energy) over runs.

    Args:
        results (dict): Dictionary containing training run results.
        coefficient_name (str): One of "gini", "elbow", or "energy".
        x_axis (str): X-axis value, either "num_classes" or "best_accuracy".
        layers (list, optional): List of layers to include. Defaults to all.
        matrix_types (list, optional): List of matrix types to include. Defaults to all.
        combine_runs (bool): Whether to average values for runs with the same x-axis value.
        filter_runs (list, optional): List of run names to include. Defaults to all.
        title (str, optional): Custom title for the plot.

    Example Usage:
        # Plot gini coefficients against number of classes
        plot_coefficients_over_runs(results, "gini", x_axis="num_classes", combine_runs=True)

        # Plot energy coefficients for 'layer_3' and 'query' matrix, filtering specific runs
        plot_coefficients_over_runs(results, "energy", x_axis="best_accuracy", layers=["layer_3"], matrix_types=["query"], filter_runs=["run_1", "run_2"])
    """
    x_values = defaultdict(list)  # To collect x-axis values for combining
    y_values = defaultdict(lambda: defaultdict(list))  # To collect y-values
    
    # Default title
    if not title:
        title = f"{coefficient_name.capitalize()} Coefficient over Runs"

    # Filter runs if specified
    filtered_results = {k: v for k, v in results.items() if not filter_runs or k in filter_runs}

    for run, data in filtered_results.items():
        # X-axis value: num_classes or best accuracy
        if x_axis == "num_classes":
            x_value = int(len(run.split('-')))  # Assuming class count is in the run key
        elif x_axis == "best_accuracy":
            x_value = data["Metrics"]["Best Results"]["Validation Accuracy"]
        else:
            raise ValueError("x_axis must be 'num_classes' or 'best_accuracy'")

        # Collect coefficients
        coeffs = data["Coefficients"][coefficient_name]
        svd_entries = data["SVD Diagonal Entries"]
        
        for matrix_type in matrix_types or ["query", "key", "value"]:
            for layer in layers or svd_entries.keys():
                x_values[x_value].append(x_value)
                y_values[matrix_type][layer].append(coeffs[layer][matrix_type])

    # Combine values if specified
    combined_x = sorted(x_values.keys())
    combined_y = defaultdict(lambda: defaultdict(list))

    for matrix_type, layers_data in y_values.items():
        for layer, values in layers_data.items():
            for x_val, y_val in zip(x_values.keys(), values):
                if combine_runs:
                    combined_y[matrix_type][layer].append(np.mean(y_val))
                else:
                    combined_y[matrix_type][layer].extend(y_val)

    # Plot the data
    plt.figure(figsize=(12, 6))
    for matrix_type in combined_y:
        for layer in combined_y[matrix_type]:
            print(f"{matrix_type} - {layer}: {combined_x} {combined_y[matrix_type][layer]}")
            plt.plot(
                combined_x, combined_y[matrix_type][layer],
                marker='o',
                label=f"{matrix_type} - {layer}"
            )
    
    plt.xlabel("Number of Classes" if x_axis == "num_classes" else "Validation Accuracy", fontsize=12)
    plt.ylabel(f"{coefficient_name.capitalize()} Coefficient", fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
# plot_coefficients_over_runs(results, "gini", x_axis="num_classes", combine_runs=True, filter_runs=["run_1", "run_2"])


# Function 2: Plot coefficients across layers or runs
def plot_coefficients_across_layers_or_runs(
    results, 
    coefficient_name, 
    mode="layers", 
    matrix_types=None, 
    layers=None, 
    runs=None, 
    combine_runs=True,
    title=None
):
    """
    Plot intrinsic dimensionality coefficients across layers or runs.

    Args:
        results (dict): Dictionary containing training run results.
        coefficient_name (str): One of "gini", "elbow", or "energy".
        mode (str): "layers" to plot across layers or "runs" to plot across runs.
        matrix_types (list, optional): List of matrix types to include. Defaults to all.
        layers (list, optional): Layers to include when mode="runs".
        runs (list, optional): List of runs to include when mode="runs".
        combine_runs (bool): Whether to average values for runs with the same key.
        title (str, optional): Custom title for the plot.

    Example Usage:
        # Plot gini coefficients across layers
        plot_coefficients_across_layers_or_runs(results, "gini", mode="layers")

        # Plot energy coefficients for layer_3 across different runs
        plot_coefficients_across_layers_or_runs(results, "energy", mode="runs", layers=["layer_3"], runs=["run_1", "run_2"], combine_runs=False)
    """
    if not title:
        title = f"{coefficient_name.capitalize()} Coefficient Across {mode.capitalize()}"

    plt.figure(figsize=(12, 6))

    if mode == "layers":
        for matrix_type in matrix_types or ["query", "key", "value"]:
            y_values = []
            layers = results[list(results.keys())[0]]["SVD Diagonal Entries"].keys()
            for layer in layers:
                values = [
                    results[run]["Coefficients"][coefficient_name][layer][matrix_type]
                    for run in results
                ]
                y_values.append(np.mean(values) if combine_runs else values)

            plt.plot(layers, y_values, marker='o', label=f"{matrix_type}")

    elif mode == "runs":
        if not runs:
            runs = results.keys()
        for layer in layers or ["layer_0"]:
            for matrix_type in matrix_types or ["query", "key", "value"]:
                y_values = [
                    results[run]["Coefficients"][coefficient_name][layer][matrix_type]
                    for run in runs
                ]
                plt.plot(runs, y_values, marker='o', label=f"{matrix_type} - {layer}")

    else:
        raise ValueError("mode must be 'layers' or 'runs'")

    plt.xlabel("Layers" if mode == "layers" else "Runs", fontsize=12)
    plt.ylabel(f"{coefficient_name.capitalize()} Coefficient", fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()


import matplotlib.pyplot as plt

def plot_best_accuracy_vs_classes(results, filter_runs=None, title="Best Accuracy vs Number of Classes"):
    """
    Plots the best accuracy over the number of classes for a dataset.

    Args:
        results (dict): Dictionary containing training run results.
        filter_runs (list, optional): List of run names to include. If None, all runs are included.
        title (str, optional): Title of the plot.

    Example Usage:
        # Plot all runs
        plot_best_accuracy_vs_classes(results)

        # Plot specific filtered runs
        plot_best_accuracy_vs_classes(results, filter_runs=["run_10", "run_20"])
    """
    # Initialize x (num_classes) and y (best_accuracy) lists
    num_classes_list = []
    best_accuracy_list = []

    # Filter results if filter_runs is provided
    filtered_results = {k: v for k, v in results.items() if not filter_runs or k in filter_runs}

    for run, data in filtered_results.items():
        # Extract number of classes from the run name
        try:
            num_classes = int(len(run.split('-')))  # Assuming the run name ends with the number of classes
        except ValueError:
            print(f"Warning: Could not parse number of classes from run '{run}'. Skipping.")
            continue

        # Extract best validation accuracy
        best_accuracy = data["Metrics"]["Best Results"]["Validation Accuracy"]

        # Append to the lists
        num_classes_list.append(num_classes)
        best_accuracy_list.append(best_accuracy)

    # Sort by number of classes for clean plotting
    sorted_indices = sorted(range(len(num_classes_list)), key=lambda i: num_classes_list[i])
    num_classes_list = [num_classes_list[i] for i in sorted_indices]
    best_accuracy_list = [best_accuracy_list[i] for i in sorted_indices]

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(num_classes_list, best_accuracy_list, marker='o', linestyle='-', color='b')
    plt.title(title, fontsize=16)
    plt.xlabel("Number of Classes", fontsize=12)
    plt.ylabel("Best Validation Accuracy", fontsize=12)
    plt.grid(True)
    plt.show()

