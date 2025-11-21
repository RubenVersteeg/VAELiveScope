import torch
import joblib.numpy_pickle as joblib
import numpy as np

def pad_specs(input, target_length, labels, removal_of_female=True):
    if removal_of_female:
        indices = [0 + i for i in range(len(input))]
        indices_to_remove = []
        #remove the indices and specs in the input that are below the target length and remove the labels that are 2 or greater
        for i in range(len(input)):
            if len(input[i][1]) > target_length or labels[i] >= 2:
                indices_to_remove.append(i)
        #print number of females to remove
        print(f'Removing {len(indices_to_remove)} female usvs from the dataset.')

        indices_after_removal = indices.copy()
        for i in indices_to_remove:
            indices_after_removal.remove(i)

        #create symetric padding until the target length is reached
        padded_specs = np.zeros((len(indices_after_removal), np.size(input[0], 0), target_length), dtype=np.float32)
        for i in range(len(indices_after_removal)):
            current_length = np.size(input[indices_after_removal[i]],1)
            zero_padding_length = target_length - current_length

            zero_padded_spec = np.zeros((np.size(input[indices_after_removal[i]],0), target_length))
            start_index = int((target_length - current_length) // 2)
            zero_padded_spec[:, start_index:start_index+current_length] = input[indices_after_removal[i]]
            padded_specs[i] = zero_padded_spec

        labels_after_removal = labels[indices_after_removal]
        return padded_specs, indices, indices_after_removal, labels_after_removal
    else:
        indices = [0 + i for i in range(len(input))]
        indices_after_removal = indices.copy()
        #remove the indices and specs in the input that are below the target length
        for i in range(len(input)):
            if len(input[i][1]) > target_length:
                indices_after_removal.remove(i)

        #create symetric padding until the target length is reached
        padded_specs = np.zeros((len(indices_after_removal), np.size(input[0], 0), target_length), dtype=np.float32)
        for i in range(len(indices_after_removal)):
            current_length = np.size(input[indices_after_removal[i]],1)
            zero_padding_length = target_length - current_length

            zero_padded_spec = np.zeros((np.size(input[indices_after_removal[i]],0), target_length))
            start_index = int((target_length - current_length) // 2)
            zero_padded_spec[:, start_index:start_index+current_length] = input[indices_after_removal[i]]
            padded_specs[i] = zero_padded_spec
            
        labels_after_removal = labels[indices_after_removal]
        return padded_specs, indices, indices_after_removal, labels_after_removal
    
def pad_specs_pups(input, target_length, labels, removal_of_mother = True):
    if removal_of_mother:
        indices = [0 + i for i in range(len(input))]
        indices_to_remove = []
        #remove indices mother1, mother2, pupx
        for i in range(len(input)):
            if len(input[i][1]) > target_length or labels[i] == 6 or labels[i] == 7 or labels[i] == 11:
                indices_to_remove.append(i)
        #print number of females to remove
        print(f'Removing {len(indices_to_remove)} mother and pupx usvs from the dataset.')


        indices_after_removal = indices.copy()
        for i in indices_to_remove:
            indices_after_removal.remove(i)

        #create symetric padding until the target length is reached
        padded_specs = np.zeros((len(indices_after_removal), np.size(input[0], 0), target_length), dtype=np.float32)
        for i in range(len(indices_after_removal)):
            current_length = np.size(input[indices_after_removal[i]],1)
            zero_padding_length = target_length - current_length

            zero_padded_spec = np.zeros((np.size(input[indices_after_removal[i]],0), target_length))
            start_index = int((target_length - current_length) // 2)
            zero_padded_spec[:, start_index:start_index+current_length] = input[indices_after_removal[i]]
            padded_specs[i] = zero_padded_spec

        labels_after_removal = labels[indices_after_removal]
        return padded_specs, indices, indices_after_removal, labels_after_removal
    else:
        indices = [0 + i for i in range(len(input))]
        indices_after_removal = indices.copy()
        #remove the indices and specs in the input that are below the target length
        for i in range(len(input)):
            if len(input[i][1]) > target_length:
                indices_after_removal.remove(i)

        #create symetric padding until the target length is reached
        padded_specs = np.zeros((len(indices_after_removal), np.size(input[0], 0), target_length), dtype=np.float32)
        for i in range(len(indices_after_removal)):
            current_length = np.size(input[indices_after_removal[i]],1)
            zero_padding_length = target_length - current_length

            zero_padded_spec = np.zeros((np.size(input[indices_after_removal[i]],0), target_length))
            start_index = int((target_length - current_length) // 2)
            zero_padded_spec[:, start_index:start_index+current_length] = input[indices_after_removal[i]]
            padded_specs[i] = zero_padded_spec
            
        labels_after_removal = labels[indices_after_removal]
        return padded_specs, indices, indices_after_removal

def pad_spectograms_sorted(input, target_length):
    #create symetric padding until the target length is reached
    padded_specs = np.zeros((len(input), np.size(input[0], 0), target_length), dtype=np.float32)
    for i in range(len(input)):
        current_length = np.size(input[i],1)
        zero_padding_length = target_length - current_length

        zero_padded_spec = np.zeros((np.size(input[i],0), target_length))
        start_index = int((target_length - current_length) // 2)
        zero_padded_spec[:, start_index:start_index+current_length] = input[i]
        #normalization
        # zero_padded_spec = np.log(zero_padded_spec + 1e-10) ##### remove after testing again is not supposed to be heres
        padded_specs[i] = zero_padded_spec

    return padded_specs

import cupy as cp

def pad_spectogram_cupy(input, target_length):
    #create symetric padding until the target length is reached
    padded_spec = cp.zeros(( cp.size(input, 0), target_length), dtype=cp.float32)
    current_length = cp.size(input,1)
    zero_padded_spec = cp.zeros((cp.size(input,0), target_length))
    start_index = int((target_length - current_length) // 2)
    zero_padded_spec[:, start_index:start_index+current_length] = input
    padded_spec = zero_padded_spec

    return padded_spec