import torch
import joblib.numpy_pickle as joblib
import numpy as np
import cupy as cp

def normalize_specs(input, max_value_per_spec=True):
    #Log normalize the spectrograms
    specs = np.log(input + 1e-10)  # Adding a small value to avoid log(0)
    # specs = input ### remove after testing again is not supposed to be here
    # Normalize the spectrograms. Can be chosen to normalize per spectrogram or globally.
    if max_value_per_spec:
        for i in range(specs.shape[0]):
            specs[i] = specs[i] / np.max(specs[i])

    else:
        specs = specs / np.max(specs)

    return specs

def normalize_spec_cupy(input, max_value_per_spec=True, max_value=1000):
    #Log normalize the spectrograms
    spec = cp.log(input + 1e-10)  # Adding a small value to avoid log(0)
    # specs = input ### remove after testing again is not supposed to be here
    # Normalize the spectrograms. Can be chosen to normalize per spectrogram or globally.
    if max_value_per_spec:
            spec = spec / cp.max(spec)

    else:
        spec = spec / max_value

    return spec