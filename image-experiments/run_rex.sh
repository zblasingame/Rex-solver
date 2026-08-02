#!/bin/bash

python scripts/image_editing_rex.py \
    --num_inference_steps 100 \
    --freeze_step 0.5 \
    --num_images 100 \
    --guidance 3.0 \
    --tableau euler \
    --zeta 0.999 \
    --prediction_type noise \
    --eps 0.0002 \
    --device 1 \
    --save_dir results/image_edits/mask_runs/rex_euler/100
