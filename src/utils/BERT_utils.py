import argparse
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import numpy as np

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT with optional LoRA on various datasets.")

    # Dataset and preprocessing
    parser.add_argument(
        "--dataset",
        choices=["arxiv"],
        required = True,
        help="Dataset to use: 'arxiv' (default: arxiv)",
    )
    # Training parameters
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Max sequence length of input."
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
    """Extract singular values from LoRA weights in the attention layers of BERT."""
    # Access BERT encoder layers through `bert.encoder.layer`
    encoder_layers = model.bert.encoder.layer  # Update for BertForSequenceClassification
    num_layers = len(encoder_layers)

    svd_results = {}

    for layer_idx in tqdm(range(num_layers), desc="Extracting LoRA SVD"):
        attention = encoder_layers[layer_idx].attention
        lora_layers = {
            'query': attention.self.query,
            'key': attention.self.key,
            'value': attention.self.value
        }

        layer_svd = {}
        for name, lora_layer in lora_layers.items():
            # Check if the layer has LoRA weights
            if hasattr(lora_layer, 'lora_A') and hasattr(lora_layer, 'lora_B'):
                # Combine LoRA A and B matrices
                lora_A = lora_layer.lora_A['default'].weight
                lora_B = lora_layer.lora_B['default'].weight
                lora_weight = torch.mm(lora_B, lora_A)

                # Perform SVD
                U, S, Vt = torch.svd(lora_weight.detach())
                layer_svd[name] = S.cpu().numpy().tolist()

        svd_results[f'layer_{layer_idx}'] = layer_svd

    return svd_results