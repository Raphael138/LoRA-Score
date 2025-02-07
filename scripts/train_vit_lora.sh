#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="ViT_training.py"

# Define dataset to be used
DATASET_1="stanford-dogs"
DATASET_2="oxford-pet"
DATASET=$DATASET_1

# Define number of classes combinations and total number of classes
if [ "$DATASET" = "$DATASET_1" ]; then
    classes_combo=(2 10 20 40 60 80 100)
    TOTAL_CLASSES=120
else
    classes_combo=(2 5 10 15 20 25 30)
    TOTAL_CLASSES=37
fi

# # Running script for all classes
# echo "Running all classes"
# python $PYTHON_SCRIPT --dataset $DATASET

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
        
        echo "Running for $num_classes random classes: $random_classes"
        
        # Run the Python script with the generated classes
        python $PYTHON_SCRIPT --dataset $DATASET --classes $random_classes --use_lora
        
        # Add a small delay to avoid overlaps (optional)
        sleep 1
    done
done
