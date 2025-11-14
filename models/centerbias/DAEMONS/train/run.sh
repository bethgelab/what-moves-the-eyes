#!/usr/bin/env bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --partition=2080-preemptable-galvani,bethge
#SBATCH --cpus-per-task=8

set -e

scontrol show job "$SLURM_JOB_ID"

r3 checkout . "$SCRATCH/job"
cd "$SCRATCH/job"

if [ -f "output/done" ]; then
    echo "Job is done already."
    exit 0
fi

mkdir -p "$SCRATCH/home"

export TQDM_MININTERVAL=30

if srun bash run_inner.sh; then
    echo "completed" > "output/done"
else
    echo "Job failed. Remove the output/done marker to try again."
    echo "failed" > "output/done"
fi
