import torch
import joblib.numpy_pickle as joblib
import numpy as np
import os
#this python file contains a function to make labels for the USV data. There can be chosen if lables should be made for 
#each emitter or labels are made for KO, WT and female.
def make_labels(input,label_type='emitter'):
    if label_type == 'emitter':
        emitters = input
        labels = []
        emitters_already_seen = []
        i = 0
        emitter_library = {}
        for emitter in emitters:
            if emitter not in emitters_already_seen:
                labels.append(i)
                emitters_already_seen.append(emitter)
                emitter_library[emitter] = i
                i += 1

            else:
                labels.append(emitters_already_seen.index(emitter))
        labels = np.array(labels, dtype=np.int64)


    elif label_type == 'strain':
        labels = []
        #library of strain names and their corresponding labels is experiment specific
        # index 0 is WT, index 1 is KO, index 2 is female, index 3 is for everything else#
        strain_library = {  'M096': 1,
                            'F102': 2,
                            'M771': 0,
                            'M103': 0,
                            'F232': 2,
                            'M104': 1,
                            'F100': 2,
                            'F231': 2,
                            'F486': 2,
                            'M770': 1,
                            'M098': 0,
                            'F234': 2,
                            'M480': 1,
                            'M479': 0,
                            'track_0': 3,
                            'M773': 1,
                            'M582': 0,
                            'M064': 1,
                            'M074': 0,
                            'M075': 0,
                            'M528': 0,
                            'M530': 0,
                            'M069': 0,
                            'M078': 1,
                            'F881': 2,
                            'M063': 1,
                            'M067': 0,
                            'M148': 1,
                            'M581': 1,
                            'F884': 2,
                            'F885': 2,
                            'F880': 2,
                            'M146': 1,
                            'M793': 1,
                            'FD881': 3,
                            'M702': 0,
                            'M701': 1,
                            'M692': 0,
                            'M697': 1,
                            'M698': 0,
                            'M699': 1,}
        for emitter in input:
            if emitter in strain_library:
                labels.append(strain_library[emitter])
            else:
                labels.append(4)  # 4 for everything else that is not in the library
        labels = np.array(labels, dtype=np.int64)
        emitter_library = strain_library
    
    return labels, emitter_library
