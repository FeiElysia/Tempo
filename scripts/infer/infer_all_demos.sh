#!/bin/bash
# ==================================================================
# Tempo-6B: Quick Demo Inference Script
# This script sequentially runs all examples defined in examples/demo_cases.json
# ==================================================================

echo "Starting Tempo-6B Demo Inference Pipeline..."
python run_demo_cases.py
echo "All demo cases completed successfully!"