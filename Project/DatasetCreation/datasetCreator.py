
import numpy as np
import os
import librosa
from my_utils.videoProcessing import *
from my_utils.processCsvFile import *

#--------------------------------------------------
import my_utils.feature_extraction_functions as ft
from my_utils.feature_extraction_functions import *
from Configurations.constants import *
#--------------------------------------------------

labeling_files_path = LABELING_FILES_PATH
path_to_root = "../../"

def construct(paths, dataset_HyP, features_file, file_rows):

    sampling_frequency = dataset_HyP['HyP_data_global']['sampling_frequency']

    total_ball_hits = 0
    total_non_ball_hits = 0

    # to balance the dataset
    non_ball_hit_vd_idx = [35, 36, 37, 38, 47, 52, 53, 54]

    video_files = np.array(list(os.listdir(path_to_root + paths[0])))
    for i in range(len(video_files)):

        v = Video(path_to_root + paths[0], video_files[i])
        video = v.get_file()
        vd_idx = int(video_files[i].split(".")[0].split("_")[1][0:])
        print("Processing data from video number ", vd_idx, "...")

        audio_path = path_to_root + paths[1] + "/" + "AUDIO_" + str(vd_idx) + ".wav"
        video.audio.write_audiofile(audio_path, fps=sampling_frequency)
        y, sr = librosa.load(audio_path, sr=None)


        nr_ball_hits, nr_non_ball_hits = get_data_features(y, vd_idx, dataset_HyP, file_rows, non_ball_hit_vd_idx)

        total_ball_hits += nr_ball_hits
        total_non_ball_hits += nr_non_ball_hits

    features_file.write_lines_on_file(file_rows)

    print("====================================================================================================")
    print("\n\nTotal ball hits: ", total_ball_hits)
    print("Total non ball hits: ", total_non_ball_hits)



def get_data_features(data, vd_index, dataset_HyP, file_rows, non_ball_hit_vd_idx):

    nr_segments = dataset_HyP['HyP_data_global']['number_of_segments']
    nr_shifted_samples = dataset_HyP['HyP_data_global']['number_of_shifted_samples']
    nr_samples_per_segment = nr_shifted_samples

    nr_ball_hits = 0
    nr_non_ball_hits = 0

    list_of_features_result = list()

    # Iterating through the json list
    for i in dataset_HyP['HyP_data_feature']:

        params = {}
        funcao = getattr(ft, i['feature_HyP']['function']['function_name'])
        params['data'] = data

        params['event_length'] =                dataset_HyP['HyP_data_global']['event_length']
        params['sampling_frequency'] =          dataset_HyP['HyP_data_global']['sampling_frequency']
        params['number_of_shifted_samples'] =   dataset_HyP['HyP_data_global']['number_of_shifted_samples']
        params['number_of_segments'] =          dataset_HyP['HyP_data_global']['number_of_segments']

        for p in i['feature_HyP']['function']['function_params']:
            for nome_variavel, valor_variavel in p.items():
                params[nome_variavel] = valor_variavel

        feature_result = funcao(params)

        list_of_features_result.append(feature_result)

    dataset_size = len(list_of_features_result[0])
    print("dataset_size = ", dataset_size)
    print("total number of lines = ", len(feature_result))
    for j in range(dataset_size): # 'j' is the index of the line. 'len(feature_result)' is the total number of lines
        feature_arr = []

        # verify if it's ball hit
        ini_idx = j * nr_shifted_samples
        final_idx = j * nr_shifted_samples + (nr_segments*nr_samples_per_segment)

        is_ball_hit = False
        is_ball_hit = get_range_label(ini_idx, final_idx, vd_index)

        for feat in list_of_features_result:
            feature_arr = np.append(feature_arr, feat[j])

        feature_arr = np.ravel(feature_arr)

        # label in the array
        feature_arr = np.append(feature_arr, [is_ball_hit * 1])  # 'feature_arr' corresponds to 1 row in the dataset


        # keep the features data on array
        if (is_ball_hit and vd_index not in non_ball_hit_vd_idx) or \
                ((not is_ball_hit) and vd_index in non_ball_hit_vd_idx):
            file_rows.append(feature_arr)

        # counter of ball and non ball hits
        if is_ball_hit and vd_index not in non_ball_hit_vd_idx:
            nr_ball_hits += 1

        elif (not is_ball_hit) and vd_index in non_ball_hit_vd_idx:
            nr_non_ball_hits += 1

    return nr_ball_hits, nr_non_ball_hits


def get_range_label(ini_idx, fin_idx, video_number):

    # read the .csv file
    file = CsvFile(path_to_root + LABELING_FILES_PATH + "/labeling_" + str(video_number) + ".csv", "r")

    # select intended columns from the file
    columns = file.get_columns(np.array([0, 3]))

    # iterate over the array and verify if its a ball hit
    for i in range(len(columns)):
        column = columns[i]
        first_sample = int(column[2])
        last_sample = int(column[3])

        condition_1 = column[0] == "racket"

        condition_2 = fin_idx >= first_sample and ini_idx <= last_sample

        if condition_1 and condition_2:
            return True

    return False



