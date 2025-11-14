#!/bin/bash

set -e

singularity exec \
        --home "$SCRATCH/home" \
        --bind $(pwd) \
        --pwd $(pwd) \
        --bind $SCRATCH \
        --bind $R3_REPOSITORY \
        --env PYTHONPATH=what-moves-the-eyes:pysaliency:combined-gaze-datasets \
        container.sif \
        python run.py
