#!/bin/bash

python scripts/ablations_rex.py \
    --num_inference_steps 84 \
    --freeze_step 0.6 \
    --num_images 100 \
    --guidance 3.0 \
    --tableau euler \
    --zeta 0.999 \
    --prediction_type noise \
    --eps 0.0 \
    --device 1 \
    --variants full no_coupling no_exp no_reparam \
    --save_dir results/ablations/rex_euler50