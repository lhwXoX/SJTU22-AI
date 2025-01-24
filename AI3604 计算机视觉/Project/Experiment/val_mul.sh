#!/bin/bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES="0"

# Define variables
arch="vit_b"  # Change this value as needed
finetune_type="lora"
dataset_name="Flare"  # Assuming you set this if it's dynamic
targets='multi_all'
# Construct the checkpoint directory argument
dir_checkpoint="SAM_Flare_Lora=4"

# Run the Python script
python val_mul_noprompt.py \
    -if_warmup True \
    -finetune_type "$finetune_type" \
    -arch "$arch" \
    -if_update_encoder True \
    -if_encoder_lora_layer True \
    -if_decoder_lora_layer True \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint"\
    -targets "$targets"