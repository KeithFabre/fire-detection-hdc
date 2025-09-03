#!/bin/bash

# Custom execution order
echo "Running experiments in custom order..."
echo "======================================"

# Run specific files in specific order

python3 "train_vgg16_hdc_random_projection.py" # ok
python3 "train_vgg16_hdc_record_based.py"      # ok
python3 "train_vgg16_hdc_adapthd.py"           # rodar de novo com batch 32
python3 "train_vgg16_hdc_neuralhd.py"          # ok
python3 "train_vgg16_hdc_onlinehd.py"          # ok

#python3 "train.py"                             # ok

echo "Selected experiments completed!" 