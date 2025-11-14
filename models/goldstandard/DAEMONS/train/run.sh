#!/usr/bin/env bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --partition=2080-preemptable-galvani,bethge
#SBATCh --mem=300G
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00
# S BATCH --exclude=galvani-cn123

set -e

scontrol show job "$SLURM_JOB_ID"

r3 checkout . "$SCRATCH/job"
cd "$SCRATCH/job"

if [ -f "output/done" ]; then
    if [[ $(< output/done ) != "restart" ]]; then
        echo "Job is done already."
        exit 0
    else
        echo "Job failed previously. Restart requested, will proceed"
    fi
fi

mkdir -p "$SCRATCH/home"

if [[ -z "${SLURM_STEP_ID}" ]]; then
    # running in sbatch
    STEP_CMD="srun bash"
else
    # allows to run job interactively within srun
    STEP_CMD="bash"
fi

export TQDM_MININTERVAL=30
export TQDM_DELAY=180

if $STEP_CMD run_inner.sh; then
    echo "completed" > "output/done"
else
    echo "Job failed. Remove the output/done marker to try again."
    echo "failed" > "output/done"
fi
