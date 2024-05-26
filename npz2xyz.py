#!/usr/bin/env python

import numpy as np

npz_file_path = 'archive/p_255.npz'
xyz_file_path = 'fluid.xyz'

data = np.load(npz_file_path)
arr = data['arr_0']

if arr.shape[1] != 3:
    raise ValueError(
        'The input data does not have the correct shape of (N, 3)')

arr_f32 = arr.astype(np.float32)

with open(xyz_file_path, 'wb') as f:
    arr_f32.tofile(f)

print(f"the .xyz file is saved as: {xyz_file_path}")
