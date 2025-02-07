import numpy as np

def gini_coefficient(x):
    """
    Calculate the Gini coefficient of a numpy array.
    
    Args:
    x (numpy.ndarray): Array of numeric values
    
    Returns:
    float: Gini coefficient
    """
    # Handle empty array
    if len(x) == 0:
        return 0.0
    
    # Sort the array
    x = np.sort(x)
    n = len(x)
        
    return  1 - 2*np.sum(x)/n

def energy_ratio_test_count(x, energy_threshold=0.95):
    """
    Implements the Energy Ratio Test and returns the count of significant values of x.
    
    Parameters:
    - x (list or np.ndarray): List of floating point values.
    - energy_threshold (float): Threshold for cumulative energy (default is 0.95).
    
    Returns:
    - count (int): Number of floating point values in the list contributing to the specified energy threshold.
    """
    # Compute cumulative energy
    x = np.array(x)
    cumulative_energy = np.cumsum(x ** 2) / np.sum(x ** 2)
    
    # Find the count of significant singular values
    significant_indices = np.where(cumulative_energy >= energy_threshold)[0]
    count = significant_indices[0] + 1 if significant_indices.size > 0 else len(x)
    
    return count

def elbow_method_count(singular_values):
    """
    Implements the Elbow Method and returns the count of singular values up to the elbow point.
    
    Parameters:
    - singular_values (list or np.ndarray): List of singular values.
    
    Returns:
    - count (int): Number of singular values up to the elbow point.
    """
    singular_values = np.array(singular_values)
    
    # Normalize the indices and values for better scaling
    indices = np.arange(len(singular_values))
    x = indices / indices.max()
    y = singular_values / singular_values.max()
    
    # Calculate the line from the first to the last point
    start = np.array([0, y[0]])
    end = np.array([1, y[-1]])
    line_vector = end - start
    
    # Compute perpendicular distance from each point to the line
    point_vectors = np.stack([x, y], axis=1) - start
    line_length = np.linalg.norm(line_vector)
    distances = np.abs(np.cross(line_vector, point_vectors)) / line_length
    
    # The elbow is the point with the maximum distance
    elbow_index = np.argmax(distances)
    
    # Count is elbow index + 1 (inclusive of the index)
    count = elbow_index + 1
    
    return count

def add_intrinsic_dimension_coeffs(results):
    """
    Adds intrinsic dimensionality coefficients (Gini, Energy, Elbow) for SVD diagonal entries to the dataset.

    Args:
        results (dict): Dictionary of runs containing "SVD Diagonal Entries".

    Returns:
        dict: Updated results dictionary with added "Coefficients".
    """
    updated_results = {}

    for run, data in results.items():
        # Initialize the Coefficients dictionary
        coefficients = {"gini": {}, "energy": {}, "elbow": {}}
        
        svd_entries = data["SVD Diagonal Entries"]  # Singular values
        
        # Iterate through layers and matrix types
        for layer, matrices in svd_entries.items():
            coefficients["gini"][layer] = {}
            coefficients["energy"][layer] = {}
            coefficients["elbow"][layer] = {}
            
            for matrix_type, singular_values in matrices.items():
                # Compute Gini coefficient
                gini = gini_coefficient(singular_values)
                
                # Compute Energy Ratio Test
                energy_count = energy_ratio_test_count(singular_values, energy_threshold=0.95)
                
                # Compute Elbow Method
                elbow_count = elbow_method_count(singular_values)
                
                # Store the results
                coefficients["gini"][layer][matrix_type] = gini
                coefficients["energy"][layer][matrix_type] = energy_count
                coefficients["elbow"][layer][matrix_type] = elbow_count
        
        # Add the computed coefficients to the results
        updated_data = data.copy()
        updated_data["Coefficients"] = coefficients
        updated_results[run] = updated_data

    return updated_results
