# Importing libraries
from tqdm import tqdm
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForImageClassification,
    ViTFeatureExtractor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import evaluate
import numpy as np
import warnings
import random 
import os
from collections import Counter

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import pandas as pd
import json

# Importing the arg parser
from utils import parse_args, gather_metrics, perform_lora_svd

# Set warnings to ignore to keep output clean
warnings.filterwarnings('ignore')

# Parse the arguments
args = parse_args()

# Printing arg info
print(f"Training on {args.dataset} with clamping: {args.clamp_min_classes} and shuffling: {args.shuffle_label_ratio}")

# Dataset selection
if args.dataset == "mnist":
    dataset = load_dataset("mnist")
    num_classes = 10
    label_column_name = 'label'
    image_col_name = "image"

elif args.dataset == "oxford-pet":
    dataset = load_dataset("visual-layer/oxford-iiit-pet-vl-enriched")
    num_classes = 37
    label_column_name = 'label_breed'
    image_col_name = "image"

elif args.dataset == "stanford-dogs":
    dataset = load_dataset("amaye15/stanford-dogs")
    num_classes = 120
    label_column_name = 'label'
    image_col_name = "pixel_values"

else:
    raise ValueError("Currently not supported -> You can add them now")

# Setting the clamp on the size of the dataset
min_num_classes = args.clamp_min_classes if args.clamp_min_classes else num_classes

# Creating val/train split
dataset = dataset["train"].train_test_split(test_size=0.15, shuffle=True, seed=1)
train_dataset = dataset['train']
val_dataset = dataset['test']

# Preprocessing for the labels -> Only necessary for oxford-pet, not mnist or stanford-dogs
if args.dataset == "oxford-pet":
    label_encoder = LabelEncoder()

    def label_preprocessing(dataset):
        # Fit the encoder on the string labels and transform them to integer labels
        label_encoder.fit(dataset[label_column_name])
        encoded_labels = label_encoder.transform(dataset[label_column_name])

        # Add the encoded labels as a new column in the dataset
        return dataset.add_column('label', encoded_labels)

    # Apply preprocessing
    train_dataset = label_preprocessing(train_dataset)
    val_dataset = label_preprocessing(val_dataset)

# Obtain clamp value for later
def get_clamp_values(dataset, min_num_classes):
    # Count labels in the training and testing datasets
    label_counts = Counter(dataset['label'])

    sorted_counts = sorted(label_counts.values())
    return sum(sorted_counts[:min_num_classes])

clamp_value = get_clamp_values(train_dataset, min_num_classes)
print(f"Clamp values: {clamp_value}")

# Filter classes if specified
if args.classes:
    selected_classes = args.classes

    def filter_classes(batch):
        return batch['label'] in selected_classes

    train_dataset = train_dataset.filter(filter_classes)
    val_dataset = val_dataset.filter(filter_classes)

    # Update num_classes to reflect the number of selected classes
    num_classes = len(selected_classes)

    # Preprocessing for the labels -> Once filtered the labels need to be set between (0, num(classes)-1)
    label_encoder = LabelEncoder()

    def label_preprocessing(dataset):
        # Fit the encoder only on the filtered labels
        label_encoder.fit(selected_classes)
        # Transform the dataset labels
        dataset = dataset.map(lambda batch: {'label': label_encoder.transform([batch['label']])[0]})
        return dataset

    # Apply preprocessing
    train_dataset = label_preprocessing(train_dataset)
    val_dataset = label_preprocessing(val_dataset)

else:
    selected_classes = [i for i in range(num_classes)]

# Clamping size of dataset
def clamp_dataset(dataset, num_classes, clamp_value):
    # Min number of samples per class    
    per_class_lim = clamp_value // num_classes

    # Group samples by class
    sample_by_class = {}
    for sample in tqdm(dataset, desc="Clamping"):
        cls = sample['label']
        if cls not in sample_by_class:
            sample_by_class[cls] = []

        if len(sample_by_class[cls])<per_class_lim:
            sample_by_class[cls].append(sample)

    clamped_dataset = []
    for samples in sample_by_class.values():
        clamped_dataset += samples

    # Convert back to a Dataset format
    filtered_data = {
        image_col_name: [sample[image_col_name] for sample in clamped_dataset],
        'label': [sample['label'] for sample in clamped_dataset],
    }
    return Dataset.from_dict(filtered_data)

# Clamping train_dataset if needed
if args.classes:
    train_dataset = clamp_dataset(train_dataset, num_classes, clamp_value)

# To shuffle portion of labels
def shuffle_labels(dataset, shuffle_fraction):
    # Calculate the number of labels to shuffle
    num_samples = len(dataset)
    num_to_shuffle = int(num_samples * shuffle_fraction)
    print(f"Shuffling {num_to_shuffle}/{num_samples} labels.")

    if num_to_shuffle==0:
        return dataset
    
    # Randomly select indices to shuffle
    indices_to_shuffle = random.sample(range(num_samples), num_to_shuffle)

    # Shuffle the selected labels
    shuffled_labels = [dataset[i]['label'] for i in indices_to_shuffle]
    random.shuffle(shuffled_labels)

    shuffled_dataset = {image_col_name: [], 'label': []}
    for i, sample in tqdm(enumerate(dataset), desc="Shuffling"):
        if i in indices_to_shuffle:
            new_label = shuffled_labels.pop(0)
            sample['label'] = new_label

        shuffled_dataset[image_col_name].append(sample[image_col_name])
        shuffled_dataset['label'].append(sample['label'])

    return Dataset.from_dict(shuffled_dataset)

# Shuffle 'em labels in train
train_dataset = shuffle_labels(train_dataset, args.shuffle_label_ratio)

# Preprocessing dataset to be compatible with ViT
transform = Compose([
    Resize((224, 224)),
    ToTensor()
])

# Combined function to resize, convert to RGB, and then to tensor
def preprocess_images(batch):
    batch['pixel_values'] = [transform(image.convert("RGB")) for image in batch[image_col_name]]
    if image_col_name!='pixel_values':
        del batch[image_col_name]
    return batch

# Apply resizing
train_dataset = train_dataset.map(preprocess_images, batched=True)
val_dataset = val_dataset.map(preprocess_images, batched=True)

# Decrease size of val dataset
new_size = min(int(len(train_dataset) * .1), len(val_dataset))
val_dataset = val_dataset.select(range(new_size))

# Printing info
print(f"Length of train dataset: {len(train_dataset)}")
print(f"Length of val dataset: {len(val_dataset)}")

if args.use_lora:
    layers = ["query", "key", "value"]
    target_modules = [f"vit.encoder.layer.{i}.attention.attention.{layer}" for i in range(0, 12) for layer in layers]


    # Set up LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        use_rslora=True,
    )

# Load model and tokenizer
model_name = "google/vit-base-patch16-224" 
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels = num_classes, ignore_mismatched_sizes=True)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Apply LoRA to the model
if args.use_lora:
    model = get_peft_model(model, lora_config)

# Move model to GPU
model = model.to("cuda")

# Define accuracy metric
accuracy = evaluate.load("accuracy")

# Define the compute_metrics function to calculate accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

run_name = f"ViT-{args.dataset}-{args.shuffle_label_ratio}"
training_args = TrainingArguments(
    output_dir=f"results/{run_name}",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    label_names = ["labels"],
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    max_steps=args.max_steps,
    logging_steps=20,
    eval_steps=20,
    save_steps=20,
    save_total_limit=1,
    evaluation_strategy="steps"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

######################
### Results Logger ###
######################

run_name = "-".join([str(i) for i in sorted(selected_classes)])
if args.shuffle_label_ratio > 0:
    run_name = f"{args.shuffle_label_ratio}: {run_name}"

if args.use_lora:
    # If LoRA, then extract the diagonal entries for singular value decomposition
    svd_diagonal_entries = perform_lora_svd(model)
    metrics = gather_metrics(trainer)

    data = {
        "Metrics": metrics,
        "SVD Diagonal Entries": svd_diagonal_entries
    }

    clamp_text = "_clamped" if args.clamp_min_classes else ""
    shuffle_text = "_shuffled" if args.shuffle_label_ratio>0 else ""
    file_name = f"out/vision/{args.dataset}/lora{clamp_text}{shuffle_text}_results.json"
    with open(file_name) as f:
        curr_results = json.load(f)

    curr_results[run_name] = data

    with open(file_name, "w") as f:
        json.dump(curr_results, f)

else:
    metrics = gather_metrics(trainer)

    with open("out/vision/fine-tune_metric_results.json") as f:
        curr_results = json.load(f)

    curr_results[args.dataset][run_name] = metrics

    with open("out/vision/fine-tune_metric_results.json", "w") as f:
        json.dump(curr_results, f)
