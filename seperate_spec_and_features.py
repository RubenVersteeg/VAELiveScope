import torch
import joblib.numpy_pickle as joblib
import numpy as np
#This python file contains a function to separate the spectrograms and features from the USV data.
def separate_spec_and_features(input):
    all_data = input

    spec = []
    emitter = []
    distance_to_nearest_non_emitter = []
    duration = []
    start_time_in_recording = []
    end_time_in_recording = []
    intensities = []
    intensity = []
    location = []
    main_freq = []
    recording_path = []
    for usv in all_data:
        keys_not_found = []
        # Ensure all keys are present in the usv dictionary
        keys = ['spec', 'emitter', 'distance_to_nearest_non_emitter',
                'duration', 'start_time_in_recording', 'end_time_in_recording',
                'intensities', 'intensity', 'location', 'main_freq', 'recording_path']
        for key in keys:
            if key not in usv:
                keys_not_found.append(key)
        #skip appending for the keys that are not found
        if 'spec' not in keys_not_found:
            spec.append(usv['spec'])
        if 'emitter' not in keys_not_found:
            emitter.append(usv['emitter'])
        if 'distance_to_nearest_non_emitter' not in keys_not_found:
            distance_to_nearest_non_emitter.append(usv['distance_to_nearest_non_emitter'])
        if 'duration' not in keys_not_found:
            duration.append(usv['duration'])
        if 'start_time_in_recording' not in keys_not_found:
            start_time_in_recording.append(usv['start_time_in_recording'])
        if 'end_time_in_recording' not in keys_not_found:
            end_time_in_recording.append(usv['end_time_in_recording'])
        if 'intensities' not in keys_not_found:
            intensities.append(usv['intensities'])
        if 'intensity' not in keys_not_found:
            intensity.append(usv['intensity'])
        if 'location' not in keys_not_found:
            location.append(usv['location'])
        if 'main_freq' not in keys_not_found:
            main_freq.append(usv['main_freq'])
        if 'recording_path' not in keys_not_found:
            recording_path.append(usv['recording_path'])

    return spec, emitter, distance_to_nearest_non_emitter, duration, start_time_in_recording, end_time_in_recording, intensities, intensity, location, main_freq, recording_path

def seperate_spec_and_emitter_pups(input):
    emitters = input

    #seperate spec and features
    spec = []
    emitter = []
    for i in range(len(emitters)):
        spec.append(emitters[i][1])
        emitter.append(emitters[i][0])

    return spec, emitter