#!/bin/bash

# Custom execution order
echo "Running experiments in custom order..."
echo "======================================"

# Run specific files in specific order
python3 "train_vgg16_hdc_random_projection.py"
python3 "train_vgg16_hdc_record_based.py"
python3 "train_vgg16_hdc_adapthd.py"
python3 "train_vgg16_hdc_neuralhd.py"
python3 "train_vgg16_hdc_onlinehd.py"

echo "Selected experiments completed!"