
import numpy as np
import tensorflow as tf
import librosa


def feature_RMS(params):

    data = params["data"]
    event_length = params["event_length"]
    sampling_frequency = params["sampling_frequency"]
    number_of_segments = params["number_of_segments"]
    number_of_shifted_samples = params["number_of_shifted_samples"]

    # calculate feature
    rms_feature = librosa.feature.rms(y=data, hop_length=number_of_shifted_samples)[0]

    # windowing the data
    rms_feature = tf.data.Dataset.from_tensor_slices(rms_feature)

    shift_nr = 1
    feature_result = rms_feature.window(size=number_of_segments, shift=shift_nr, drop_remainder=True)
    feature_result = feature_result.flat_map(lambda window: window.batch(number_of_segments))

    feature_result = np.array(list(feature_result.as_numpy_iterator()))

    return feature_result


def feature_Onset(params):

    data = params["data"]
    event_length = params["event_length"]
    sampling_frequency = params["sampling_frequency"]
    number_of_segments = params["number_of_segments"]
    number_of_shifted_samples = params["number_of_shifted_samples"]


    # calculate the feature
    onset_feature = librosa.onset.onset_detect(y=data, sr=sampling_frequency, hop_length=number_of_shifted_samples)
    onset_feature = generate_onset_array(onset_feature, len(data), number_of_shifted_samples)  # sets onsets indexes to 1 (where occured onset)

    # windowing the data
    onset_feature = tf.data.Dataset.from_tensor_slices(onset_feature)

    shift_nr = 1
    feature_result = onset_feature.window(size=number_of_segments, shift=shift_nr, drop_remainder=True)
    feature_result = feature_result.flat_map(lambda window: window.batch(number_of_segments))

    feature_result = np.array(list(feature_result.as_numpy_iterator()))

    return feature_result

def generate_onset_array(arr, data_size, nr_shifted_samples):
    arr_size = data_size / nr_shifted_samples
    arr_size = int(arr_size) + 1
    onset_array = np.zeros(arr_size)
    onset_array[arr] = 1

    return onset_array



def feature_Spectralflux(params):


    data = params["data"]
    event_length = params["event_length"]
    sampling_frequency = params["sampling_frequency"]
    number_of_segments = params["number_of_segments"]
    number_of_shifted_samples = params["number_of_shifted_samples"]

    # calculate feature
    specflux_feature = librosa.onset.onset_strength(y=data, sr=sampling_frequency, hop_length=number_of_shifted_samples)

    # windowing the data
    specflux_feature = tf.data.Dataset.from_tensor_slices(specflux_feature)

    shift_nr = 1
    feature_result = specflux_feature.window(size=number_of_segments, shift=shift_nr, drop_remainder=True)
    feature_result = feature_result.flat_map(lambda window: window.batch(number_of_segments))

    feature_result = np.array(list(feature_result.as_numpy_iterator()))

    return feature_result




def feature_BER(params):

    data = params["data"]
    sampling_frequency = params["sampling_frequency"]
    number_of_segments = params["number_of_segments"]
    number_of_shifted_samples = params["number_of_shifted_samples"]
    FRAME_SIZE = number_of_shifted_samples
    split_frequency = params["split_frequency"]

    # Extract Spectrograms
    data_spectrogram = librosa.stft(y=data, n_fft=FRAME_SIZE, hop_length=number_of_shifted_samples)


    # calculate feature
    split_frequency_bin = calculate_split_frequency_bin(data_spectrogram, split_frequency, sampling_frequency)

    # Move to the power spetrogram
    power_spectrogram = np.abs(data_spectrogram) ** 2
    power_spectrogram = power_spectrogram.T

    band_energy_ratio = []
    # Calculate BER for each frame (BER is a frame-based feature)
    for frequencies_in_frame in power_spectrogram:
        sum_power_low_frequencies = np.sum(frequencies_in_frame[:split_frequency_bin])
        sum_power_high_frequencies = np.sum(frequencies_in_frame[split_frequency_bin:])
        ber_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        band_energy_ratio.append(ber_current_frame)

    # windowing the data
    band_energy_ratio = tf.data.Dataset.from_tensor_slices(band_energy_ratio)

    shift_nr = 1
    feature_result = band_energy_ratio.window(size=number_of_segments, shift=shift_nr, drop_remainder=True)
    feature_result = feature_result.flat_map(lambda window: window.batch(number_of_segments))

    feature_result = np.array(list(feature_result.as_numpy_iterator()))

    return feature_result


def calculate_split_frequency_bin(spectrogram, split_frequency, sample_rate):

    frequency_range = sample_rate / 2
    frequency_delta_per_bin = frequency_range / spectrogram.shape[0]
    split_frequency_bin = np.floor(split_frequency / frequency_delta_per_bin)
    return int(split_frequency_bin)

