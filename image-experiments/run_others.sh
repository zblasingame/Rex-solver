#!/bin/bash

python scripts/image_editing_rex.py \
    --num_inference_steps 167 \
    --num_images 100 \
    --freeze_step 0.6 \
    --guidance 3.0 \
    --sampler_type edict \
    --edict_p 0.93 \
    --device 0 \
    --save_dir results/image_edits/mask_runs/edict/100