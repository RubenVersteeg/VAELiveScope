import cupy as cp
import joblib.numpy_pickle as numpy_pickle
import matplotlib.pyplot as plt
import warnings
import time
#in this file i will write functions to extract the typical features from mouse USVs. All these features will be
#extracted from a spectogram of the USV. The spectogram is a 2D array where the x-axis is time and the y-axis is frequency.
#The spectograms are already created by the protocol of the lab. All these feature extraction functions will be implemented in a live 
#setting. This means that the user will be able to see the spectogram and the features extracted from it in real time.

def get_spectral_line_values(spectogram):
    #get the spectra lines of the spectogram and corrects for noise
    spectogram = cp.array(spectogram)
    spectral_line_index = cp.argmax(spectogram, axis=0)
    spectral_line = cp.zeros(len(spectogram[0]))
    for i in range(len(spectogram[0])):
        spectral_line[i] = spectogram[spectral_line_index[i],i]
    #correct edge effects
    Nsteps = len(spectral_line)
    for i in [0,1] + [Nsteps-2, Nsteps-1]:
        if i in {0,1}:
            cmean = spectral_line[2]
            if abs(1 -spectral_line[i] / cmean) > 0.05:
                spectral_line[i] = cmean
            continue
        elif i in {Nsteps-2, Nsteps - 1}:
            cmean = spectral_line[Nsteps-3]
            if abs(1 -spectral_line[i] / cmean) > 0.05:
                spectral_line[i] = cmean
            continue
    #average over neighbors if deviation is too strong
    if len(spectral_line) > 3:
        for i in range(2, len(spectral_line) - 1):
            cmean = (spectral_line[i-1] + spectral_line[i] + spectral_line[i+1])/3
            BigLocalChange = abs(1 - spectral_line[i] / cmean) 
            SimilarToNeigh = spectral_line[i-1]/spectral_line[i+1]
            #local change is smaller than 5% and difference between neigbouring bins is smaller than 10%
            if BigLocalChange > 0.05 and SimilarToNeigh < 0.1:
                spectral_line[i] = cmean
    
    return spectral_line_index, spectral_line

def get_spectral_line_index(spectogram):
     #get the spectra lines of the spectogram and corrects for noise
    spectogram = cp.array(spectogram)
    spectral_line_index = cp.argmax(spectogram, axis=0)
    
    
    return spectral_line_index

def correct_frequency_voc(spectral_line_index, min_freq = 0, sample_rate =200000, nb_freq_bins = 129):
    """This function corrects the frequency values of the vocalisation. The frequency values are the y-axis of the spectogram. The correction is done by taking the mean of the frequency values of the spectogram. The corrected frequency values are a measure of the corrected frequency of the USV. The min_freq and max_freq are the minimum and maximum frequency of the spectogram. In our analysis
    the spectogram is displayed from 0 to 100 kHz. The nb_freq_bins is the number of frequency bins in the spectogram. 
    In our analysis this is 129."""
    return spectral_line_index * ((sample_rate/2) - min_freq) / nb_freq_bins + min_freq

def index_voc(data):
    """This function returns the index of the vocalisation"""
    max_index = len(data) 
    return cp.arange(0, max_index,1)

def duration_voc(spectral_line_index, sample_rate = 200000, time_bin_size = 8192/36):
    """Sample rate is the number of samples per second at which the audio was recorded. In our setup this is 200000.
    Time bin size is the number of effective samples per time bin. In our created spectograms this is 8192/36. Spectrograms are created 
    every 8192 samples with an overlap of 32 samples. Therefore in 8192 samples there are 36 time bins. Vocalisations are later
    represented as a series of time bins. This can be higher or lower than the 36 time bins, where the original spectrograms were created
    from."""
    return cp.array([len(spectral_line_index) * time_bin_size / sample_rate])


def mean_frequency_voc(spectral_line_index):
    """This function calculates the mean frequency of the spectogram. The mean frequency is calculated by taking the mean of the 
    frequency values of the spectogram. The frequency values are the y-axis of the spectogram. The mean frequency is a measure of the 
    average frequency of the USV. The min_freq and max_freq are the minimum and maximum frequency of the spectogram. In our analysis
    the spectogram is displayed from 0 to 100 kHz. The nb_freq_bins is the number of frequency bins in the spectogram. 
    In our analysis this is 129."""
    return cp.array([cp.mean(spectral_line_index)])

def min_frequency_voc(spectral_line_index):
    """This function calculates the minimum frequency of the vocalisation. The minimum frequency is calculated by taking the minimum of the 
    frequency values of the vocalisation. The frequency values are the y-axis of the spectogram. The minimum frequency is a measure of the 
    lowest frequency of the USV. The min_freq and max_freq are the minimum and maximum frequency of the spectogram. In our analysis
    the spectogram is displayed from 0 to 100 kHz. The nb_freq_bins is the number of frequency bins in the spectogram. 
    In our analysis this is 129."""
    return cp.array([cp.min(spectral_line_index)])

def max_frequency_voc(spectral_line_index):
    """This function calculates the maximum frequency of the vocalisation. The maximum frequency is calculated by taking the maximum of the 
    frequency values of the vocalisation. The frequency values are the y-axis of the spectogram. The maximum frequency is a measure of the 
    highest frequency of the USV. The min_freq and max_freq are the minimum and maximum frequency of the spectogram. In our analysis
    the spectogram is displayed from 0 to 100 kHz. The nb_freq_bins is the number of frequency bins in the spectogram. 
    In our analysis this is 129."""
    return cp.array([cp.max(spectral_line_index)])

def bandwidth_voc(spectral_line_index):
    """This function calculates the bandwidth of the vocalisation. The bandwidth is calculated by taking the difference between the maximum
    and minimum frequency of the vocalisation. The frequency values are the y-axis of the spectogram. The bandwidth is a measure of the 
    range of frequencies of the USV. The min_freq and max_freq are the minimum and maximum frequency of the spectogram. In our analysis
    the spectogram is displayed from 0 to 100 kHz. The nb_freq_bins is the number of frequency bins in the spectogram. 
    In our analysis this is 129."""
    return cp.array(max_frequency_voc(spectral_line_index) - min_frequency_voc(spectral_line_index))

def starting_frequency_voc(spectral_line_index):
    """This function calculates the starting frequency of the vocalisation. The starting frequency is calculated by taking the first frequency
    value of the vocalisation. The frequency values are the y-axis of the spectogram. The starting frequency is a measure of the 
    frequency of the USV at the beginning. The min_freq and max_freq are the minimum and maximum frequency of the spectogram. In our analysis
    the spectogram is displayed from 0 to 100 kHz. The nb_freq_bins is the number of frequency bins in the spectogram. 
    In our analysis this is 129."""
    return cp.array([spectral_line_index[0]])

def stopping_frequency_voc(spectral_line_index):
    """This function calculates the stopping frequency of the vocalisation. The stopping frequency is calculated by taking the last frequency
    value of the vocalisation. The frequency values are the y-axis of the spectogram. The stopping frequency is a measure of the 
    frequency of the USV at the end. The min_freq and max_freq are the minimum and maximum frequency of the spectogram. In our analysis
    the spectogram is displayed from 0 to 100 kHz. The nb_freq_bins is the number of frequency bins in the spectogram. 
    In our analysis this is 129."""
    return cp.array([spectral_line_index[-1]])

def directionality_voc(spectral_line_index):
    """This function calculates the directionality of the vocalisation. """
    coeffs = cp.polyfit(cp.arange(len(spectral_line_index)), spectral_line_index, 1)
    slope = coeffs[0]
    return cp.array([slope])

# def variance_voc(spectral_line_index):
#     """This function calculates the variance of the vocalisation. The variance is calculated by taking the variance of the frequency values of the vocalisation.
#     The frequency values are the y-axis of the spectogram. The variance is a measure of the
#     variation of the USV. The min_freq and max_freq are the minimum and maximum frequency of the spectogram. In our analysis. The variance
#     shows the variation of the USV per time bin"""
#     return cp.array([cp.var(spectral_line_index)])

def coefficient_of_variation_voc(spectral_line_index_frequency):
    """
    Calculates the Coefficient of Variation (StdDev/Mean) for the USV's frequency contour.
    This provides a normalized, unitless measure of overall frequency spread.
    """
    mean_freq = cp.mean(spectral_line_index_frequency)
    std_dev_freq = cp.std(spectral_line_index_frequency)
    
    # Handle the case where mean frequency is zero (shouldn't happen with USVs, but for robustness)
    if cp.isclose(mean_freq, 0.0):
        return cp.array([0.0])
        
    return cp.array([std_dev_freq / mean_freq])

def normalized_irregularity_voc(spectral_line_index_frequency):
    """
    Calculates the standard deviation of the step size normalized by the mean absolute step size.
    This is a normalized measure of local contour roughness/irregularity.
    """
    
    # Calculate the step-to-step differences
    diffs = cp.diff(spectral_line_index_frequency)
    
    # Numerator: Standard deviation of the differences (related to variance_voc_specline)
    std_dev_diffs = cp.std(diffs)
    
    # Denominator: Mean absolute difference (this is local_variability_voc)
    mean_abs_diffs = cp.mean(cp.abs(diffs))
    
    # Handle the case where the contour is perfectly flat (no change)
    if cp.isclose(mean_abs_diffs, 0.0):
        # If mean change is zero, irregularity is minimal (0/0 is indeterminate, but 0 is safe)
        return cp.array([0.0])
        
    return cp.array([std_dev_diffs / mean_abs_diffs])

# def variance_voc_specline(spectral_line_index):
#     """This function calculates the variance of the in the spectral line of the vocalisation. The variance is calculated by taking the variance of the differences between the frequency values of the vocalisation.
#     The frequency values are the y-axis of the spectogram. The variance is a measure of the
#     variation of the USV. The min_freq and max_freq are the minimum and maximum frequency of the spectogram. In our analysis. The variance
#     shows the variation of the USV per time bin
    
#     --- spectral contour irregularity ---

#     """
#     return cp.array([cp.var(cp.diff(spectral_line_index))])

def local_variability_voc(spectral_line_index):
    """This function calculates the local variability of the vocalisation. The local variability is calculated by taking the mean of the differences between the frequency values of the vocalisation.
    The frequency values are the y-axis of the spectogram. The local variability is a measure of the
    local variation of the USV. The min_freq and max_freq are the minimum and maximum frequency of the spectogram. In our analysis. The local variability
    shows the local variation of the USV per time bin"""
    return cp.array([cp.mean(cp.abs(cp.diff(spectral_line_index)))])

def nr_of_step_up(spectral_line_index, step_size = 5):
    """This function calculates the number of steps up in the vocalisation. The number of steps up is calculated by taking the number of positive differences between the frequency values of the vocalisation.
    The frequency values are the y-axis of the spectogram. The number of steps up is a measure of the
    number of steps up in the USV. The min_freq and max_freq are the minimum and maximum frequency of the spectogram. In our analysis. The number of steps up
    shows the number of steps up in the USV per time bin"""
    return cp.array([cp.sum(cp.diff(spectral_line_index) > step_size)])

def nr_of_step_down(spectral_line_index, step_size = 5):
    """This function calculates the number of steps down in the vocalisation. The number of steps down is calculated by taking the number of negative differences between the frequency values of the vocalisation.
    The frequency values are the y-axis of the spectogram. The number of steps down is a measure of the
    number of steps down in the USV. The min_freq and max_freq are the minimum and maximum frequency of the spectogram. In our analysis. The number of steps down
    shows the number of steps down in the USV per time bin"""
    return cp.array([cp.sum(cp.diff(spectral_line_index) < -step_size)])

def nr_of_peak(spectral_line_index,nr_of_steps_down,nr_of_steps_up, min_slope = 0.005, step_size = 5):
    """This function calculates the number of peaks in the vocalisation. """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cp.RankWarning)
        if nr_of_steps_down == 0 and nr_of_steps_up == 0:
            nr_of_points = len(spectral_line_index)
            polyfit_complexity = 3
            if nr_of_points >60:
                polyfit_complexity = int(nr_of_points/20)
            if polyfit_complexity > 9:
                polyfit_complexity = 9
            coeffs = cp.polyfit(cp.arange(len(spectral_line_index)), spectral_line_index, polyfit_complexity)
            poly = cp.poly1d(coeffs)
            x = cp.arange(len(spectral_line_index))
            y = poly(x)
            dy = cp.gradient(y)
            d2y = cp.gradient(dy)
            peaks = []
            for i in range(1, len(dy)-1):
                if dy[i] > min_slope and dy[i+1] < -min_slope and d2y[i] < 0:
                    peaks.append(i)
                elif dy[i] < -min_slope and dy[i+1] > min_slope and d2y[i] < 0:
                    peaks.append(i)
            return cp.array([len(peaks)])
        else:
            """Determine the number of peaks for every part of the vocalisation."""
            #find indexes at which a jump occurs
            indexes = cp.where(cp.abs(cp.diff(spectral_line_index)) > step_size)
            indexes = cp.concatenate((cp.array([-1]), indexes[0], cp.array([len(spectral_line_index)-1])))
            peaks = []
            for i in range(len(indexes)-1):
                index1 = indexes[i].item()
                index2 = indexes[i+1].item()
                #get the part of the spectral line between index1 and index2
                spectral_line_part = spectral_line_index[index1+1:index2+1]
                spectral_line_part = cp.array(spectral_line_part)
                nr_of_points = len(spectral_line_part)
                if nr_of_points < 3:
                    continue
                polyfit_complexity = 3
                if nr_of_points >60:
                    polyfit_complexity = int(nr_of_points/20)
                if polyfit_complexity > 9:
                    polyfit_complexity = 9
                coeffs = cp.polyfit(cp.arange(len(spectral_line_part)), spectral_line_part, polyfit_complexity)
                poly = cp.poly1d(coeffs)
                x = cp.arange(len(spectral_line_part))
                y = poly(x)
                dy = cp.gradient(y)
                d2y = cp.gradient(dy)
                for i in range(1, len(dy)-1):
                    if dy[i] > min_slope and dy[i+1] < -min_slope and d2y[i] < 0:
                        peaks.append(i)
                    elif dy[i] < -min_slope and dy[i+1] > min_slope and d2y[i] < 0:
                        peaks.append(i)
            lenght = len(peaks)
            return cp.array([lenght])

def nr_of_valley(spectral_line_index,nr_of_steps_down,nr_of_steps_up, min_slope = 0.005,step_size = 5):
    """This function calculates the number of valleys in the vocalisation. """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cp.RankWarning)
        if nr_of_steps_down == 0 and nr_of_steps_up == 0:
            nr_of_points = len(spectral_line_index)
            polyfit_complexity = 3
            if nr_of_points >60:
                polyfit_complexity = int(nr_of_points/20)
            if polyfit_complexity > 9:
                polyfit_complexity = 9
            coeffs = cp.polyfit(cp.arange(len(spectral_line_index)), spectral_line_index, polyfit_complexity)
            poly = cp.poly1d(coeffs)
            x = cp.arange(len(spectral_line_index))
            y = poly(x)
            dy = cp.gradient(y)
            dy_0 = dy > 0
            diffdy_0 = cp.diff(dy_0)
            diffdy = cp.diff(dy)
            d2y = cp.gradient(dy)
            valleys = []
            # for i in range(1, len(dy)-1):
            #     if dy[i] < -min_slope and dy[i+1] > min_slope and d2y[i] > 0:
            #         valleys.append(i)
            #     elif dy[i] > min_slope and dy[i+1] < -min_slope and d2y[i] > 0:
            #         valleys.append(i)
            # return cp.array([len(valleys)])
        else:
            """Determine the number of valleys for every part of the vocalisation."""
            #find indexes at which a jump occurs
            indexes = cp.where(cp.abs(cp.diff(spectral_line_index)) > step_size)
            indexes = cp.concatenate((cp.array([-1]), indexes[0], cp.array([len(spectral_line_index)-1])))
            valleys = []
            for i in range(len(indexes)-1):
                index1 = indexes[i].item()
                index2 = indexes[i+1].item()
                #get the part of the spectral line between index1 and index2
                spectral_line_part = spectral_line_index[index1+1:index2+1]
                spectral_line_part = cp.array(spectral_line_part)
                nr_of_points = len(spectral_line_part)
                if nr_of_points < 3:
                    continue
                polyfit_complexity = 3
                if nr_of_points >60:
                    polyfit_complexity = int(nr_of_points/20)
                if polyfit_complexity > 9:
                    polyfit_complexity = 9
                coeffs = cp.polyfit(cp.arange(len(spectral_line_part)), spectral_line_part, polyfit_complexity)
                poly = cp.poly1d(coeffs)
                x = cp.arange(len(spectral_line_part))
                y = poly(x)
                dy = cp.gradient(y)
                d2y = cp.gradient(dy)
                for i in range(1, len(dy)-1):
                    if dy[i] < -min_slope and dy[i+1] > min_slope and d2y[i] > 0:
                        valleys.append(i)
                    elif dy[i] > min_slope and dy[i+1] < -min_slope and d2y[i] > 0:
                        valleys.append(i)
            length = len(valleys)
            return cp.array([length])
        
def nr_of_peaks_and_valleys(spectral_line_index,nr_of_steps_down,nr_of_steps_up, min_difference = 0.1,step_size = 5):
    """This function calculates the number of valleys in the vocalisation. """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cp.RankWarning)
        if nr_of_steps_down == 0 and nr_of_steps_up == 0:
            nr_of_points = len(spectral_line_index)
            polyfit_complexity = 3
            if nr_of_points >60:
                polyfit_complexity = int(nr_of_points/20)
            if polyfit_complexity > 9:
                polyfit_complexity = 9
            coeffs = cp.polyfit(cp.arange(len(spectral_line_index)), spectral_line_index, polyfit_complexity)
            poly = cp.poly1d(coeffs)
            x = cp.arange(len(spectral_line_index))
            y = poly(x)
            dy = cp.gradient(y)
            diff_dy = cp.diff(dy)
            difference = cp.abs(diff_dy) > min_difference
            dy_0 = dy > 0
            diffdy_0 = cp.diff(dy_0)
            d2y = cp.gradient(dy)
            d2y_0 = d2y > 0
            nr_of_peaks = 0
            nr_of_valleys = 0
            for i in range(len(y)-1):
                if diffdy_0[i] == True and d2y_0[i] == False and difference[i] == True:
                    nr_of_peaks += 1
                elif diffdy_0[i] == True and d2y_0[i] == True and difference[i] == True:
                    nr_of_valleys += 1

            return cp.array([nr_of_peaks]), cp.array([nr_of_valleys])
        else:
            indexes = cp.where(cp.abs(cp.diff(spectral_line_index)) > step_size)
            indexes = cp.concatenate((cp.array([-1]), indexes[0], cp.array([len(spectral_line_index)-1])))
            nr_of_peaks = 0
            nr_of_valleys = 0
            for i in range(len(indexes)-1):
                index1 = indexes[i].item()
                index2 = indexes[i+1].item()
                #get the part of the spectral line between index1 and index2
                spectral_line_part = spectral_line_index[index1+1:index2+1]
                spectral_line_part = cp.array(spectral_line_part)
                nr_of_points = len(spectral_line_part)
                if nr_of_points < 3:
                    continue
                polyfit_complexity = 3
                if nr_of_points >60:
                    polyfit_complexity = int(nr_of_points/20)
                if polyfit_complexity > 9:
                    polyfit_complexity = 9
                coeffs = cp.polyfit(cp.arange(len(spectral_line_part)), spectral_line_part, polyfit_complexity)
                poly = cp.poly1d(coeffs)
                x = cp.arange(len(spectral_line_part))
                y = poly(x)
                x_np = cp.asnumpy(x)
                y_np = cp.asnumpy(y)
                spectral_line_part_np = cp.asnumpy(spectral_line_part)
                dy = cp.gradient(y)
                diff_dy = cp.diff(dy)
                difference = cp.abs(diff_dy) > min_difference
                dy_0 = dy > 0
                diffdy_0 = cp.diff(dy_0)
                d2y = cp.gradient(dy)
                d2y_0 = d2y > 0
                for i in range(len(y)-1):
                    if diffdy_0[i] == True and d2y_0[i] == False and difference[i] == True:
                        nr_of_peaks += 1
                    elif diffdy_0[i] == True and d2y_0[i] == True and difference[i] == True:
                        nr_of_valleys += 1
            return cp.array([nr_of_peaks]), cp.array([nr_of_valleys])

def get_usv_features(data):
    data = cp.array(data)
    spectral_line_index = get_spectral_line_index(data)
    spectral_line_index_frequency = correct_frequency_voc(spectral_line_index)
    duration = duration_voc(spectral_line_index) #0
    mean_frequency = mean_frequency_voc(spectral_line_index_frequency) #1
    min_frequency = min_frequency_voc(spectral_line_index_frequency) #2
    max_frequency = max_frequency_voc(spectral_line_index_frequency) #3
    bandwidth = bandwidth_voc(spectral_line_index_frequency)  #4
    starting_frequency = starting_frequency_voc(spectral_line_index_frequency) #5
    stopping_frequency = stopping_frequency_voc(spectral_line_index_frequency) #6
    directionality = directionality_voc(spectral_line_index_frequency) #7
    # variance = variance_voc(spectral_line_index_frequency) #8
    # variance_specline = variance_voc_specline(spectral_line_index_frequency) #9
    coefficient_of_variation = coefficient_of_variation_voc(spectral_line_index_frequency) # NEW (REPLACING variance)
    normalized_irregularity = normalized_irregularity_voc(spectral_line_index_frequency) # NEW (REPLACING variance_specline)
    local_variability = local_variability_voc(spectral_line_index_frequency) #10
    nr_of_steps_up = nr_of_step_up(spectral_line_index) #11
    nr_of_steps_down = nr_of_step_down(spectral_line_index)	 #12
    # nr_of_peaks = nr_of_peak(spectral_line_index,nr_of_steps_down,nr_of_steps_up) #13
    # nr_of_valleys = nr_of_valley(spectral_line_index,nr_of_steps_down,nr_of_steps_up) #14
    nr_of_peaks, nr_of_valleys = nr_of_peaks_and_valleys(spectral_line_index,nr_of_steps_down,nr_of_steps_up) #13,14
    # USV_features = cp.stack([duration, mean_frequency, min_frequency, max_frequency, bandwidth, starting_frequency, stopping_frequency, directionality, variance, variance_specline, local_variability, nr_of_steps_up, nr_of_steps_down, nr_of_peaks, nr_of_valleys], axis=0)
    USV_features = cp.stack([duration, mean_frequency, min_frequency, max_frequency, bandwidth, starting_frequency, stopping_frequency, directionality, coefficient_of_variation, normalized_irregularity, local_variability, nr_of_steps_up, nr_of_steps_down, nr_of_peaks, nr_of_valleys], axis=0)
    return USV_features

def get_usv_features_without_integers(data):
    data = cp.array(data)
    spectral_line_index = get_spectral_line_index(data)
    spectral_line_index_frequency = correct_frequency_voc(spectral_line_index)
    duration = duration_voc(spectral_line_index) #0
    mean_frequency = mean_frequency_voc(spectral_line_index_frequency) #1
    min_frequency = min_frequency_voc(spectral_line_index_frequency) #2
    max_frequency = max_frequency_voc(spectral_line_index_frequency) #3
    bandwidth = bandwidth_voc(spectral_line_index_frequency)  #4
    starting_frequency = starting_frequency_voc(spectral_line_index_frequency) #5
    stopping_frequency = stopping_frequency_voc(spectral_line_index_frequency) #6
    directionality = directionality_voc(spectral_line_index_frequency) #7
    # variance = variance_voc(spectral_line_index_frequency) #8
    # variance_specline = variance_voc_specline(spectral_line_index_frequency) #9
    coefficient_of_variation = coefficient_of_variation_voc(spectral_line_index_frequency) # NEW (REPLACING variance)
    normalized_irregularity = normalized_irregularity_voc(spectral_line_index_frequency) # NEW (REPLACING variance_specline)
    local_variability = local_variability_voc(spectral_line_index_frequency) #10
    # nr_of_steps_up = nr_of_step_up(spectral_line_index) #11
    # nr_of_steps_down = nr_of_step_down(spectral_line_index)	 #12
    # nr_of_peaks = nr_of_peak(spectral_line_index,nr_of_steps_down,nr_of_steps_up) #13
    # nr_of_valleys = nr_of_valley(spectral_line_index,nr_of_steps_down,nr_of_steps_up) #14
    # nr_of_peaks, nr_of_valleys = nr_of_peaks_and_valleys(spectral_line_index,nr_of_steps_down,nr_of_steps_up) #13,14
    # USV_features = cp.stack([duration, mean_frequency, min_frequency, max_frequency, bandwidth, starting_frequency, stopping_frequency, directionality, variance, variance_specline, local_variability, nr_of_steps_up, nr_of_steps_down, nr_of_peaks, nr_of_valleys], axis=0)
    USV_features = cp.stack([duration, mean_frequency, min_frequency, max_frequency, bandwidth, starting_frequency, stopping_frequency, directionality, coefficient_of_variation, normalized_irregularity, local_variability], axis=0)
    return USV_features
# columns = ['Index,Duration, Mean Frequency, Min Frequency, Max Frequency, Bandwidth, Starting Frequency, Stopping Frequency, Directionality, Variance, Variance Specline, Local Variability']
# cp.savetxt('USV_features.csv', USV_features, delimiter=',', header=','.join(columns), comments='')
#save numpy array to file with pickle
# numpy_pickle.dump(USV_features, 'USV_features_Shank3_B1_D1_usvs.jl')
#This are some basic properties, the matlab file sent by the lab contains more features. Namely the following: Marginal amplitude distribtuion
#of vocalisation in time, skew and kurtosis of the time marginal, marginal amplitude distribution of vocalisation in frequency, 
#skew and kurtosis of the frequency marginal, the wiener entropy, the pitch salience, vibrato and position in current phrase and phrase number.
#I tried determining the spectral purity, but this is not possible, because in these spectograms the median frequency for evvery time bin is zero.

    
