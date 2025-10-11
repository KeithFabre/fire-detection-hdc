#!/bin/bash

# Custom execution order
echo "Running experiments in custom order..."
echo "======================================"

# Run specific files in specific order

#python3 "random_projection.py"     # ok
#python3 "record_based.py"          # ok
python3 "classifier_adapthd.py"     # ok 
python3 "classifier_neuralhd.py"    # ok
python3 "classifier_onlinehd.py"    # ok

echo "Selected experiments completed!"