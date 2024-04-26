
import json
import copy

from Project.DatasetCreation.datasetCreator import construct as construct_dataset

#--------------------------------------------------
from my_utils.feature_extraction_functions import *
from Configurations.constants import *
from my_utils.processCsvFile import CsvFile
#--------------------------------------------------

audios_path = AUDIOS_PATH
videos_path = VIDEOS_PATH
hyp_data_scheme_file_path = HYP_DATA_SCHEME_FILE_PATH
dataset_storage_path = DATASET_STORAGE_PATH
dataset_hyp_storage_path = DATASET_HYP_STORAGE_PATH

# Current file path relative to the project root.
path_to_root = "../../"

# Opening JSON file in 'read' mode
with open(path_to_root + hyp_data_scheme_file_path, 'r') as HyP_data_file:
    # returns JSON object as a dictionary
    dataset_HyP_data_scheme = json.load(HyP_data_file)

HyP_data_file.close()

paths_train = [videos_path, audios_path]
file_rows = list()

dataset_count = 0

for v_el in dataset_HyP_data_scheme['HyP_data_global']['event_length']['values']:
    for v_sf in dataset_HyP_data_scheme['HyP_data_global']['sampling_frequency']['values']:
        for v_nss in dataset_HyP_data_scheme['HyP_data_global']['number_of_shifted_samples']['values']:
            #for v_ns in dataset_HyP_data_scheme['HyP_data_global']['number_of_segments']['values']:

                dataset_HyP = copy.deepcopy(dataset_HyP_data_scheme)
                nr_segments = round((v_el * v_sf) / v_nss)
                dataset_count += 1

                dataset_HyP['HyP_data_global']['event_length'] = v_el
                dataset_HyP['HyP_data_global']['sampling_frequency'] = v_sf
                dataset_HyP['HyP_data_global']['number_of_shifted_samples'] = v_nss
                #dataset_HyP['HyP_data_global']['number_of_segments'] = v_ns
                dataset_HyP['HyP_data_global']['number_of_segments'] = nr_segments

                HyP_data_Dataset_xxx = path_to_root + dataset_hyp_storage_path + '/HyP_data_Dataset_' + str(dataset_count) + '.json'

                # Opening JSON file in 'write' mode
                with open(HyP_data_Dataset_xxx, 'w') as HyP_data_Dataset_xxx_file:
                    json.dump(dataset_HyP, HyP_data_Dataset_xxx_file, indent=4)

                HyP_data_Dataset_xxx_file.close()

                dataset_XXX_file = dataset_storage_path + "/Dataset_" + str(dataset_count) + ".csv"

                CsvFile.remove_file(path_to_root + dataset_XXX_file)
                features_file = CsvFile(path_to_root + dataset_XXX_file, "w")

                file_rows.clear()

                construct_dataset(paths_train, dataset_HyP, features_file, file_rows)

