#!/bin/bash

# Custom execution order
echo "Running experiments in custom order..."
echo "======================================"

# Run specific files in specific order
python3 "random_projection.py"
python3 "record_based.py"
python3 "classifier_adapthd.py"
python3 "classifier_neuralhd.py"
python3 "classifier_onlinehd.py"

echo "Selected experiments completed!"