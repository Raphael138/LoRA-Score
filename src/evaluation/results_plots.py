import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# def plot_coefficients_over_runs(
#     results, 
#     coefficient_name, 
#     x_axis="num_classes", 
#     layers=None, 
#     matrix_types=None, 
#     combine_runs=True, 
#     combine_layers=False,
#     filter_runs=None, 
#     title=None
# ):
#     """
#     Plot intrinsic dimensionality coefficients (gini, elbow, energy) over runs.

#     Args:
#         results (dict): Dictionary containing training run results.
#         coefficient_name (str): One of "gini", "elbow", or "energy".
#         x_axis (str): X-axis value, either "num_classes" or "best_accuracy".
#         layers (list, optional): List of layers to include. Defaults to all layers.
#         matrix_types (list, optional): List of matrix types to include. Defaults to ["query", "key", "value"].
#         combine_runs (bool): Whether to average values for runs with the same x-axis value.
#         combine_layers (bool): Whether to average values across all layers for each matrix type.
#         filter_runs (list, optional): List of run names to include. Defaults to all runs.
#         title (str, optional): Custom title for the plot.

#     Example Usage:
#         # Plot gini coefficients averaged over layers
#         plot_coefficients_over_runs(results, "gini", combine_layers=True)

#         # Plot energy coefficients for specific layers and matrix types
#         plot_coefficients_over_runs(results, "energy", x_axis="best_accuracy", layers=["layer_3"], matrix_types=["query"], filter_runs=["run_10", "run_20"])
#     """
#     # Initialize storage for x and y values
#     combined_data = defaultdict(list)

#     # Default title
#     if not title:
#         title = f"{coefficient_name.capitalize()} Coefficient over Runs"

#     # Filter runs if specified
#     filtered_results = {k: v for k, v in results.items() if not filter_runs or k in filter_runs}

#     for run, data in filtered_results.items():
#         # X-axis value: Extract num_classes or best_accuracy
#         if x_axis == "num_classes":
#             try:
#                 x_value = int(len(run.split('-')))  # Extract class count from the run name
#             except ValueError:
#                 print(f"Warning: Could not parse 'num_classes' for run '{run}'. Skipping.")
#                 continue
#         elif x_axis == "best_accuracy":
#             x_value = data["Metrics"]["Best Results"]["Validation Accuracy"]
#         else:
#             raise ValueError("x_axis must be 'num_classes' or 'best_accuracy'")

#         # Collect coefficients
#         coeffs = data["Coefficients"][coefficient_name]
#         svd_entries = data["SVD Diagonal Entries"]

#         for matrix_type in matrix_types or ["query", "key", "value"]:
#             if combine_layers:
#                 # Average across all layers for the given matrix type
#                 layer_values = [
#                     coeffs[layer][matrix_type] for layer in svd_entries.keys()
#                 ]
#                 avg_value = np.mean(layer_values)
#                 combined_data[matrix_type].append((x_value, avg_value))
#             else:
#                 # Keep values for individual layers
#                 for layer in layers or svd_entries.keys():
#                     combined_data[(matrix_type, layer)].append((x_value, coeffs[layer][matrix_type]))

#     # Plot the data
#     plt.figure(figsize=(12, 6))

#     if combine_layers:
#         for matrix_type, values in combined_data.items():
#             # Ensure x and y values are sorted and aligned
#             values = sorted(values, key=lambda v: v[0])
#             x_vals, y_vals = zip(*values)

#             if combine_runs:
#                 unique_x_vals = sorted(set(x_vals))
#                 averaged_y_vals = [
#                     np.mean([y for x, y in values if x == unique_x])
#                     for unique_x in unique_x_vals
#                 ]
#                 x_vals, y_vals = unique_x_vals, averaged_y_vals

#             plt.plot(x_vals, y_vals, marker='o', label=f"{matrix_type} (avg across layers)")

#     else:
#         for (matrix_type, layer), values in combined_data.items():
#             # Ensure x and y values are sorted and aligned
#             values = sorted(values, key=lambda v: v[0])
#             x_vals, y_vals = zip(*values)

#             if combine_runs:
#                 unique_x_vals = sorted(set(x_vals))
#                 averaged_y_vals = [
#                     np.mean([y for x, y in values if x == unique_x])
#                     for unique_x in unique_x_vals
#                 ]
#                 x_vals, y_vals = unique_x_vals, averaged_y_vals

#             plt.plot(x_vals, y_vals, marker='o', label=f"{matrix_type} - {layer}")

#     plt.xlabel("Number of Classes" if x_axis == "num_classes" else "Validation Accuracy", fontsize=12)
#     plt.ylabel(f"{coefficient_name.capitalize()} Coefficient", fontsize=12)
#     plt.title(title, fontsize=16)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# Example usage
# plot_coefficients_over_runs(results, "gini", x_axis="num_classes", combine_runs=True, filter_runs=["run_1", "run_2"])




import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from statistics import mode, median

def plot_coefficients_over_runs(
    results, 
    coefficient_name, 
    x_axis="num_classes", 
    layers=None, 
    matrix_types=None, 
    combine_runs=True, 
    combine_layers=False,
    combine_method="average",
    filter_runs=None, 
    title=None
):
    """
    Plot intrinsic dimensionality coefficients (gini, elbow, energy) over runs.

    Args:
        results (dict): Dictionary containing training run results.
        coefficient_name (str): One of "gini", "elbow", or "energy".
        x_axis (str): X-axis value, either "num_classes" or "best_accuracy".
        layers (list, optional): List of layers to include. Defaults to all layers.
        matrix_types (list, optional): List of matrix types to include. Defaults to ["query", "key", "value"].
        combine_runs (bool): Whether to combine values for runs with the same x-axis value.
        combine_layers (bool): Whether to combine values across all layers for each matrix type.
        combine_method (str): Method to combine values. Options are "average", "median", or "mode".
        filter_runs (list, optional): List of run names to include. Defaults to all runs.
        title (str, optional): Custom title for the plot.

    Example Usage:
        # Plot gini coefficients averaged over layers
        plot_coefficients_over_runs(results, "gini", combine_layers=True, combine_method="average")

        # Plot energy coefficients for specific layers and matrix types
        plot_coefficients_over_runs(results, "energy", x_axis="best_accuracy", layers=["layer_3"], matrix_types=["query"], combine_method="median")
    """
    # Initialize storage for x and y values
    combined_data = defaultdict(list)

    # Default title
    if not title:
        title = f"{coefficient_name.capitalize()} Coefficient over Runs"

    # Filter runs if specified
    filtered_results = {k: v for k, v in results.items() if not filter_runs or k in filter_runs}

    for run, data in filtered_results.items():
        # X-axis value: Extract num_classes or best_accuracy
        if x_axis == "num_classes":
            try:
                x_value = int(len(run.split('-')))  # Extract class count from the run name
            except ValueError:
                print(f"Warning: Could not parse 'num_classes' for run '{run}'. Skipping.")
                continue
        elif x_axis == "best_accuracy":
            x_value = data["Metrics"]["Best Results"]["Validation Accuracy"]
        else:
            raise ValueError("x_axis must be 'num_classes' or 'best_accuracy'")

        # Collect coefficients
        coeffs = data["Coefficients"][coefficient_name]
        svd_entries = data["SVD Diagonal Entries"]

        for matrix_type in matrix_types or ["query", "key", "value"]:
            if combine_layers:
                # Combine across all layers for the given matrix type
                layer_values = [
                    coeffs[layer][matrix_type] for layer in svd_entries.keys()
                ]
                combined_value = combine_values(layer_values, combine_method)
                combined_data[matrix_type].append((x_value, combined_value))
            else:
                # Keep values for individual layers
                for layer in layers or svd_entries.keys():
                    combined_data[(matrix_type, layer)].append((x_value, coeffs[layer][matrix_type]))

    # Plot the data
    plt.figure(figsize=(12, 6))

    if combine_layers:
        for matrix_type, values in combined_data.items():
            # Ensure x and y values are sorted and aligned
            values = sorted(values, key=lambda v: v[0])
            x_vals, y_vals = zip(*values)

            if combine_runs:
                x_vals, y_vals = combine_x_values(x_vals, y_vals, combine_method)

            plt.plot(x_vals, y_vals, marker='o', label=f"{matrix_type} (avg across layers)")

    else:
        for (matrix_type, layer), values in combined_data.items():
            # Ensure x and y values are sorted and aligned
            values = sorted(values, key=lambda v: v[0])
            x_vals, y_vals = zip(*values)

            if combine_runs:
                x_vals, y_vals = combine_x_values(x_vals, y_vals, combine_method)

            plt.plot(x_vals, y_vals, marker='o', label=f"{matrix_type} - {layer}")

    plt.xlabel("Number of Classes" if x_axis == "num_classes" else "Validation Accuracy", fontsize=12)
    plt.ylabel(f"{coefficient_name.capitalize()} Coefficient ({combine_method.capitalize()})", fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()


def combine_values(values, method):
    """Combine a list of values based on the specified method."""
    if method == "average":
        return np.mean(values)
    elif method == "median":
        return median(values)
    elif method == "mode":
        try:
            return mode(values)
        except:
            print("Warning: Mode calculation failed due to no unique mode.")
            return np.mean(values)  # Fallback to mean
    else:
        raise ValueError("Invalid combine_method. Choose from 'average', 'median', or 'mode'.")


def combine_x_values(x_vals, y_vals, method):
    """Combine y-values for the same x-values based on the specified method."""
    unique_x_vals = sorted(set(x_vals))
    combined_y_vals = [
        combine_values([y for x, y in zip(x_vals, y_vals) if x == unique_x], method)
        for unique_x in unique_x_vals
    ]
    return unique_x_vals, combined_y_vals


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


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem  # For confidence interval
from collections import defaultdict

def plot_coefficients_with_accuracy(
    results, 
    coefficient_name, 
    accuracies_dict, 
    layers=None, 
    combine_matrices=True,
    title=None,
    ylabel=None
):
    """
    Plot averaged coefficients with confidence intervals against averaged accuracies.

    Args:
        results (dict): Dictionary containing training run results.
        coefficient_name (str): Name of the coefficient ("gini", "elbow", "energy").
        accuracies_dict (dict): Dictionary where keys are class counts and values are lists of accuracies.
        layers (list, optional): Layers to include. Defaults to all layers.
        combine_matrices (bool): If True, averages key, query, and value coefficients. If False, plots them separately.
        title (str, optional): Title for the plot.
        ylabel (str, optional): Label for the y-axis. Defaults to the coefficient name.

    Example:
        accuracies = {10: [0.85, 0.86, 0.87], 20: [0.80, 0.81, 0.82]}
        plot_coefficients_with_accuracy(results, "gini", accuracies, combine_matrices=False)
    """
    # Initialize storage for averaged coefficients
    valid_class_counts = []
    mean_accuracies = []
    coefficients_by_class = defaultdict(lambda: defaultdict(list))  # To store coefficients for each class

    # Step 1: Validate class counts and align them with results
    for class_count, accuracies in accuracies_dict.items():
        # Check if this class count exists in results
        relevant_runs = [run for run in results if len(run.split('-')) == class_count]
        if relevant_runs:
            valid_class_counts.append(class_count)
            mean_accuracies.append(np.mean(accuracies))  # Average accuracies for this class
            for run in relevant_runs:
                coeffs = results[run]["Coefficients"][coefficient_name]
                svd_entries = results[run]["SVD Diagonal Entries"]
                for matrix_type in ["query", "key", "value"]:
                    for layer in layers or svd_entries.keys():
                        coefficients_by_class[class_count][matrix_type].append(coeffs[layer][matrix_type])
        else:
            print(f"Warning: No results for class count {class_count}. Skipping it.")

    # Step 2: Compute averaged coefficients and confidence intervals
    combined_coefficients = []
    combined_conf_intervals = []

    matrix_coefficients = defaultdict(list)  # For separate query/key/value plotting

    for class_count in valid_class_counts:
        if combine_matrices:
            # Combine across all matrix types
            all_values = []
            for matrix_type in ["query", "key", "value"]:
                all_values.extend(coefficients_by_class[class_count][matrix_type])
            if all_values:
                combined_coefficients.append(np.mean(all_values))
                combined_conf_intervals.append(sem(all_values) * 1.96)  # 95% confidence interval
        else:
            # Store separately for each matrix type
            for matrix_type in ["query", "key", "value"]:
                values = coefficients_by_class[class_count][matrix_type]
                if values:
                    matrix_coefficients[matrix_type].append((np.mean(values), sem(values) * 1.96))

    # Step 3: Plotting
    plt.figure(figsize=(10, 6))

    if combine_matrices:
        plt.errorbar(
            mean_accuracies, 
            combined_coefficients, 
            yerr=combined_conf_intervals, 
            fmt='o-', 
            capsize=5, 
            label="Combined Coefficients"
        )
    else:
        for matrix_type, data in matrix_coefficients.items():
            y_means, y_cis = zip(*data)
            plt.errorbar(
                mean_accuracies, 
                y_means, 
                yerr=y_cis, 
                fmt='o-', 
                capsize=5, 
                label=f"{matrix_type.capitalize()} Coefficients"
            )

    # Labels and Title
    plt.xlabel("Mean Validation Accuracy", fontsize=12)
    plt.ylabel(ylabel if ylabel else f"Average {coefficient_name.capitalize()} Coefficient", fontsize=12)
    plt.title(title if title else f"{coefficient_name.capitalize()} Coefficient vs Validation Accuracy", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Show plot
    plt.show()

