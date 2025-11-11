#!/bin/bash

# Example usage script for Multi-turn Dialogue Evaluation Tool
# This script demonstrates different ways to run the evaluation

echo "Multi-turn Dialogue Evaluation Tool - Example Usage"
echo "=================================================="

# Basic global evaluation
echo "1. Running basic global evaluation..."
python score_inference.py \
    --mode global \
    --categories context_awareness,helpfulness \
    --reason \
    --limit 5

echo -e "\n2. Running local evaluation for individual turns..."
python score_inference.py \
    --mode local \
    --categories local_image_text_consistency,text_quality \
    --reason \
    --limit 3

echo -e "\n3. Running comprehensive evaluation (both modes)..."
python score_inference.py \
    --mode both \
    --categories all \
    --reason \
    --model gpt-4o \
    --temperature 0.2 \
    --limit 2

echo -e "\n4. Quick evaluation without reasoning..."
python score_inference.py \
    --mode global \
    --categories all \
    --no-reason \
    --limit 1

echo -e "\nExample runs completed!"
echo "Check the output directory for results." 