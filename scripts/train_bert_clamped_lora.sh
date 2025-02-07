#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="BERT_training.py"

# Set the dataset
DATASET_1="arxiv"
DATASET="$1"

# Define number of classes combinations and total number of classes
if [ "$DATASET" = "$DATASET_1" ]; then
    classes_combo=(20 40 60 80 100 120)
    TOTAL_CLASSES=140
    min_classes=20
else
    echo "Invalid argument. Please use another dataset."
    exit 1
fi

# Function to generate a random combination of classes
generate_random_classes() {
    local num_classes=$1
    shuf -i 0-$(($TOTAL_CLASSES - 1)) -n $num_classes | tr '\n' ' ' | sed 's/,$//'
}

# Loop through the desired combinations
for i in {1..10}; do
    for num_classes in ${classes_combo[@]}; do
        # Generate random classes
        random_classes=$(generate_random_classes $num_classes)
                
        # Run the Python script with the generated classes
        python $PYTHON_SCRIPT --dataset $DATASET --classes $random_classes --use_lora --lora_rank 768 --clamp_min_classes $min_classes
        
        # Add a small delay to avoid overlaps (optional)
        sleep 1
    done
done
