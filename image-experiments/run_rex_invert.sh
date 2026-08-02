#!/bin/bash

python scripts/inversion.py \
    --num_inference_steps 50 \
    --sampler_type rex \
    --n_samples 100 \
    --tableau euler \
    --zeta 0.999 \
    --prediction_type data \
    --freeze_step 1 \
    --eps 0.0002 \
    --device 0 \
    --save_dir results/inversions/rex_euler/50
