from tqdm import tqdm
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, TaskType, get_peft_model
import evaluate
import numpy as np
import warnings
import random
import os
import json
from collections import Counter

# Importing the arg parser
from utils import gather_metrics
from BERT_utils import parse_args, perform_lora_svd

# Suppress warnings
warnings.filterwarnings('ignore')

args = parse_args()

# Dataset selection
if args.dataset == "arxiv":
    dataset = load_dataset("csv", data_files="data/arxiv_filtered.csv")
    num_classes = 140
    label_column_name = "categories"
    text_column_name = "abstract"
else:
    raise ValueError("Currently not supported")

# Setting the clamp on the size of the dataset
min_num_classes = args.clamp_min_classes if args.clamp_min_classes else num_classes

# Creating val/train split
dataset = dataset['train'].train_test_split(test_size=0.15, shuffle=True, seed=1)
train_dataset = dataset['train']
val_dataset = dataset['test']

# Preprocessing for the labels -> Only necessary for oxford-pet and arxiv, not mnist or stanford-dogs
if args.dataset == "arxiv":
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

def get_clamp_values(dataset, min_num_classes):
    label_counts = Counter(dataset['label'])

    sorted_counts = sorted(label_counts.values())
    clamp_value = sum(sorted_counts[:min_num_classes])
    return clamp_value

clamp_value = get_clamp_values(train_dataset, min_num_classes)
print(f"Clamp value: {clamp_value}")


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
    # Count labels in the training and testing datasets
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
        text_column_name: [sample[text_column_name] for sample in clamped_dataset],
        'label': [sample['label'] for sample in clamped_dataset],
    }
    return Dataset.from_dict(filtered_data)

if args.classes:
    # Clamping train_dataset
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

    shuffled_dataset = {text_column_name: [], 'label': []}
    for i, sample in tqdm(enumerate(dataset), desc="Shuffling"):
        if i in indices_to_shuffle:
            new_label = shuffled_labels.pop(0)
            sample['label'] = new_label

        shuffled_dataset[text_column_name].append(sample[text_column_name])
        shuffled_dataset['label'].append(sample['label'])

    return Dataset.from_dict(shuffled_dataset)

# Shuffle 'em labels in train
train_dataset = shuffle_labels(train_dataset, args.shuffle_label_ratio)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(batch):
    tokenized = tokenizer(
        batch[text_column_name],
        padding="max_length",
        truncation=True,
        max_length=args.max_seq_length
    )

    return tokenized

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Decrease size of val dataset
new_size = min(int(len(train_dataset) * .1), len(val_dataset))
val_dataset = val_dataset.select(range(new_size))

# Printing info
print(f"Length of dataset: {len(train_dataset)}")
print(f"Length of validation dataset: {len(val_dataset)}")

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(train_dataset.unique('label'))
)

# Apply LoRA
if args.use_lora:
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "key", "value"],  # BERT attention modules
        lora_dropout=args.lora_dropout,
        task_type=TaskType.SEQ_CLS,  # Task is sequence classification
    )
    model = get_peft_model(model, lora_config)

# Move model to GPU
model = model.to("cuda")

# Define evaluation metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

run_name = f"BERT-{args.dataset}-{args.shuffle_label_ratio}"
training_args = TrainingArguments(
    output_dir=f"results/{run_name}",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    max_steps=args.max_steps,
    logging_steps=20,
    eval_steps=20,
    save_steps=20,
    save_total_limit=1,
    evaluation_strategy="steps",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset= val_dataset,
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
    file_name = f"out/text/{args.dataset}/lora{clamp_text}{shuffle_text}_results.json"
    with open(file_name) as f:
        curr_results = json.load(f)

    curr_results[run_name] = data

    with open(file_name, "w") as f:
        json.dump(curr_results, f)

else:
    metrics = gather_metrics(trainer)

    with open("out/text/fine-tune_metric_results.json") as f:
        curr_results = json.load(f)

    curr_results[args.dataset][run_name] = metrics

    with open("out/text/fine-tune_metric_results.json", "w") as f:
        json.dump(curr_results, f)
