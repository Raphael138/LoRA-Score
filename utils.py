import argparse
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import numpy as np
import os
from scipy.stats import linregress

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT with optional LoRA on various datasets.")

    # Dataset and preprocessing
    parser.add_argument(
        "--dataset",
        choices=["mnist", "stanford-dogs", "oxford-pet", "FGVC-Aircraft", "caltech-101", "food101", "flowers-102"],
        required = True,
        help="Dataset to use: 'mnist' or 'oxford' (default: oxford)",
    )

    # Classes selection
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="List of classes to train on (e.g., --classes 1 7 29)."
    )

    # Whether to clamp size of dataset
    parser.add_argument(
        "--clamp_min_classes",
        type=int,
        default=None,
        help="If clamping the size of the dataset, the minimum number of classes used in the training (default: None (no clamping))."
    )

    # Whether to shuffle the labels
    parser.add_argument(
        "--shuffle_label_ratio",
        type=float,
        default=0.,
        help=" If you want to shuffle a portion of the labels in the test dataset, the ratio to do so(default: 0 (no shuffling))."
    )

    # LoRA configuration
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Apply LoRA to the model."
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=500,
        help="Rank parameter for LoRA (default: 500)."
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=500,
        help="Alpha parameter for LoRA -> suggested to be equal to r(default: 500)."
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout parameter for LoRA (default: 0.1)."
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Per-device batch size (default: 8)."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=50,
        help="Per-device batch size (default: 8)."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 2)."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum training steps (default: 500)."
    )

    return parser.parse_args()

def perform_lora_svd(model):
    """Extract singular values from LoRA weights across all layers."""
    encoder_layers = model.base_model.model.vit.encoder.layer
    num_layers = len(encoder_layers)
    
    svd_results = {}
    
    for layer_idx in tqdm(range(num_layers)):
        attention = encoder_layers[layer_idx].attention.attention
        lora_layers = {
            'query': attention.query,
            'key': attention.key,
            'value': attention.value
        }
        
        layer_svd = {}
        for name, lora_layer in lora_layers.items():
            # Combine LoRA A and B matrices
            lora_A = lora_layer.lora_A['default'].weight
            lora_B = lora_layer.lora_B['default'].weight
            lora_weight = torch.mm(lora_B, lora_A)
            
            # Perform SVD
            U, S, Vt = torch.svd(lora_weight.detach())
            layer_svd[name] = S.cpu().numpy().tolist()
        
        svd_results[f'layer_{layer_idx}'] = layer_svd
    
    return svd_results

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

def gather_metrics(trainer):
    """Given a tranformer trainer, return dictionary with training and validation loss 
       as well as the accuracy."""

    log_history = trainer.state.log_history

    # Initialize lists for training and validation metrics
    steps = []
    training_loss = []
    validation_loss = []
    validation_accuracy = []

    # Extract relevant metrics
    for log in log_history:
        if 'loss' in log and 'step' in log:  # Training logs
            steps.append(log['step'])
            training_loss.append(log['loss'])
        elif 'eval_loss' in log and 'step' in log:  # Validation logs
            validation_loss.append(log['eval_loss'])
            validation_accuracy.append(log['eval_accuracy'])

    # Create a DataFrame
    data = {
        "Step": steps,
        "Training Loss": training_loss,
        "Validation Loss": validation_loss,
        "Validation Accuracy": validation_accuracy,
        "Best Results": {
            "Training Loss": min(training_loss),
            "Validation Loss": min(validation_loss),
            "Validation Accuracy": max(validation_accuracy),
        }
    }

    return data

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

