# pylint: disable=missing-module-docstring,invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order

import argparse
import glob
import logging
import os

import torch
import yaml
from deepgaze_pytorch.config import config_schema
from deepgaze_pytorch.loading import expand_config
from deepgaze_pytorch.training import run_training_part

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('PIL').setLevel(logging.INFO)
logging.getLogger('numba').setLevel(logging.INFO)
logging.getLogger('h5py').setLevel(logging.INFO)

if not torch.cuda.is_available():
    raise ValueError("CUDA not available")

root_directory = os.path.dirname(os.path.realpath(__file__))
print(root_directory)
print(os.getcwd())

task_id = os.environ.get('SLURM_ARRAY_TASK_ID', None)
if task_id is not None:
    crossval_fold_number_default = int(task_id)
    print("Running cross validation fold", crossval_fold_number_default)
else:
    crossval_fold_number_default = None


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-part', default=None, type=str)
    parser.add_argument('--crossval-fold-number', default=crossval_fold_number_default, type=int)
    parser.add_argument('--sub-experiment-no', default=None, type=int)
    parser.add_argument('--sub-experiment', default=None, type=str)
    parser.add_argument('--no-training', default=False, action='store_true')
    parser.add_argument('--no-evaluation', default=False, action='store_true')

    args = parser.parse_args()

    return args


args = parse_arguments()

print("Arguments", args)

if args.sub_experiment is not None:
    new_root = os.path.join(root_directory, args.sub_experiment)
    if not os.path.isdir(new_root):
        raise ValueError("Invalid sub experiment", args.sub_experiment)
    root_directory = new_root
    print("Running sub experiment", root_directory)

elif args.sub_experiment_no is not None:

    sub_experiment_candidates = glob.glob(os.path.join(root_directory, f'experiment{args.sub_experiment_no:04d}*'))

    if not sub_experiment_candidates:
        raise ValueError("No subexperiment with number", args.sub_experiment_no)
    if len(sub_experiment_candidates) > 1:
        raise ValueError("Too many candidates with number", args.sub_experiment_no)

    root_directory, = sub_experiment_candidates
    print("Running sub experiment", root_directory)


config = yaml.safe_load(open(os.path.join(root_directory, 'config.yaml')))

config = config_schema.validate(config)
print(yaml.safe_dump(config))
config_schema.validate(config)


config = expand_config(config)

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

for training_part in config['training']['parts']:
    if args.training_part is not None and training_part['name'] != args.training_part:
        print("Skipping part", args.training_part)
        continue
    run_training_part('output/', training_part, config, args=args)
