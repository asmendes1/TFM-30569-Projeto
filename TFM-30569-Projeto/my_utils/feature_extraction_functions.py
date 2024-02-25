
import numpy as np
import tensorflow as tf
import librosa
from my_utils.processCsvFile import *



def test_(params):
    print("Ola test_.")
    return params["num1"] + params["num2"]


def feature_RMS(params):
    print("========== Function feature_RMS ==========")

    data = params["data"]
    nr_groups = params["nr_segments"]
    nr_samples_per_group = params["hop_length"]
    nr_shifted_samples = params["shifted_samples"]

    # calculate feature
    rms_feature = librosa.feature.rms(y=data, hop_length=nr_samples_per_group)[0]

    # windowing the data
    rms_feature = tf.data.Dataset.from_tensor_slices(rms_feature)

    shift_nr = int(nr_shifted_samples / nr_samples_per_group)
    f2 = rms_feature.window(size=nr_groups, shift=shift_nr, drop_remainder=True)
    f2 = f2.flat_map(lambda window: window.batch(nr_groups))
    #------ (a partir daqui é código da função get_data_features)
    f2 = np.array(list(f2.as_numpy_iterator()))

    print("========== End of function feature_RMS ==========")
    return f2


def feature_Onset(params):
    print("========== Function feature_Onset ==========")

    data = params["data"]
    sample_rate = params["sampling_rate"]
    nr_groups = params["nr_segments"]
    nr_samples_per_group = params["hop_length"]
    nr_shifted_samples = params["shifted_samples"]

    # calculate feature
    onset_feature = librosa.onset.onset_detect(y=data, sr=sample_rate, hop_length=nr_samples_per_group)
    onset_feature = generate_onset_array(onset_feature, len(data), nr_samples_per_group)  # sets onsets indexes to 1 (where occured onset)

    # windowing the data
    onset_feature = tf.data.Dataset.from_tensor_slices(onset_feature)

    shift_nr = int(nr_shifted_samples / nr_samples_per_group)
    f1 = onset_feature.window(size=nr_groups, shift=shift_nr, drop_remainder=True)
    f1 = f1.flat_map(lambda window: window.batch(nr_groups))
    # ------ (a partir daqui é código da função get_data_features)
    f1 = np.array(list(f1.as_numpy_iterator()))

    print("========== End of function feature_Onset ==========")
    return f1

def generate_onset_array(arr, data_size, spg):
    arr_size = data_size / spg  # nr of onsets
    arr_size = int(arr_size) + 1
    onset_array = np.zeros(arr_size)
    onset_array[arr] = 1

    return onset_array



def feature_Spectralflux(params):
    print("========== Function feature_Specrtalflux ==========")

    data = params["data"]
    sample_rate = params["sampling_rate"]
    nr_groups = params["nr_segments"]
    nr_samples_per_group = params["hop_length"]
    nr_shifted_samples = params["shifted_samples"]

    # calculate feature
    specflux_feature = librosa.onset.onset_strength(y=data, sr=sample_rate, hop_length=nr_samples_per_group)

    # windowing the data
    specflux_feature = tf.data.Dataset.from_tensor_slices(specflux_feature)

    shift_nr = int(nr_shifted_samples / nr_samples_per_group)
    f3 = specflux_feature.window(size=nr_groups, shift=shift_nr, drop_remainder=True)
    f3 = f3.flat_map(lambda window: window.batch(nr_groups))
    # ------ (a partir daqui é código da função get_data_features)
    f3 = np.array(list(f3.as_numpy_iterator()))

    print("========== End of function feature_Specrtalflux ==========")
    return f3



def feature_BER(params):
    print("========== Function feature_BER ==========")

    data = params["data"]
    sample_rate = params["sampling_rate"]
    nr_groups = params["nr_segments"]
    HOP_SIZE = params["hop_length"] # nr_samples_per_group
    FRAME_SIZE = nr_groups * HOP_SIZE

    # Extract Spectrograms
    data_spectrogram = librosa.stft(y=data, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    print ("data_spectrogram.shape (o 1º valor é o número de níveis do espectrograma, e é isto que calculamos na variável 'frequency_range'): ", data_spectrogram.shape)

    data_spectrogram_transpose = data_spectrogram.T

    # calculate feature
    split_frequency_bin = calculate_split_frequency_bin(data_spectrogram, 2000, sample_rate)

    # Move to the power spetrogram
    power_spectrogram = np.abs(data_spectrogram) ** 2
    power_spectrogram = power_spectrogram.T

    band_energy_ratio = []
    # Calculate BER for each frame
    for frequencies_in_frame in power_spectrogram:
        sum_power_low_frequencies = np.sum(frequencies_in_frame[:split_frequency_bin])
        sum_power_high_frequencies = np.sum(frequencies_in_frame[split_frequency_bin:])
        ber_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        band_energy_ratio.append(ber_current_frame)

    print("========== End of function feature_BER ==========")
    return np.array(band_energy_ratio)

def calculate_split_frequency_bin(spectrogram, split_frequency, sample_rate):

    frequency_range = sample_rate / 2
    frequency_delta_per_bin = frequency_range / spectrogram.shape[0]
    split_frequency_bin = np.floor(split_frequency / frequency_delta_per_bin)
    return int(split_frequency_bin)

