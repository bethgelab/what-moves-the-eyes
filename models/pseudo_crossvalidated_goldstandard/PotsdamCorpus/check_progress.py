import sys

import h5py

filename = sys.argv[1]

with h5py.File(filename, mode='r') as f:
    done_count = len(f.keys())
    total_count = 90
    print(f"Done {done_count} out of {total_count} tasks.")