
import json
import copy

from Project.DatasetCreation.datasetCreator import construct as construct_dataset

#--------------------------------------------------
from my_utils.feature_extraction_functions import *
#--------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =================================  constants ================================================
# Nota 1: as constantes PATH representam o caminho relativamente à raíz do projecto (i.e. a partir da raiz).
#       (ou seja, onde elas foram usadas, deverá ser indicado à priori o caminho até à raiz (e.g. '../../')).
#       (isto assenta no pressuposto de que, quem usa a constante sabe a sua própria localização (relativamente à raiz),
#       mas a constante não pode adivinhar a localização de quem a usará).
# Nota 2: sempre que o caminho (PATH) for para um ficheiro específico, a designação termina em 'FILE_PATH'.
#           Se a desegnação terminar apenas com 'PATH', então é porque o caminho é para uma directoria.
AUDIOS_PATH = "Data/audios"
VIDEOS_PATH = "Data/videos"
HyP_data_FILE_PATH = "Hyperparameters/Dataset_HyP_structure.json"
DATASET_STORAGE_PATH = "Project/DatasetStorage"
HyP_data_DATASET_PATH = 'Project/DatasetStorage/Datasets_HyP'

# Caminho do ficheiro atual relativamente à raiz do projecto. (se eu colocar o presente ficheiro noutra directoria, basta verificar esta variável)
path_to_root = "../../"



# --------------------------------------- main --------------------------------------------------------------------

# Ler a estrutura de hiper-parâmetros (HyP_Data)
# Opening JSON file in 'read' mode
with open(path_to_root + HyP_data_FILE_PATH, 'r') as HyP_data_file:
    # returns JSON object as a dictionary
    dataset_HyP_global = json.load(HyP_data_file)

HyP_data_file.close()

paths_train = [VIDEOS_PATH, AUDIOS_PATH]
file_rows = list()

# construir um dataset para cada valor de 'N' (número de segmentos em que a janela (de 0.5seg) é dividida)
# NOTA: Variando o 'N' (nog), o hopLength (spr) terá de ser ajustado, para que o eventLength se mantenha
hopLength = [4096, 2048, 1024]
for i in hopLength:
    #dataset_HyP = {}

    dataset_HyP = copy.deepcopy(dataset_HyP_global)

    n = round((0.5*44100)/i) # N -> NOG -> number_of_segments

    dataset_XXX_file = DATASET_STORAGE_PATH + "/Dataset_N_" + str(n) + ".csv"

    # Modificar a estrutura de hiper-parâmetros para criar um HyP_data_Dataset_xxx.json
    dataset_HyP['event_length'] = dataset_HyP['event_length']['init']
    dataset_HyP['sampling_rate'] = dataset_HyP['sampling_rate']['init']
    dataset_HyP['number_of_segments'] = n           # dataset_HyP['number_of_segments']['init']
    dataset_HyP['samples_per_segment'] = i          # dataset_HyP['samples_per_segment']['init']
    dataset_HyP['number_of_shifted_samples'] = i # dataset_HyP['number_of_shifted_samples']['init'] # dataset_HyP['number_of_shifted_samples'] = 3000 * dataset_HyP['samples_per_segment']


    # Remover uma 'feature', ou seja, remover um dos valores da lista "Features" da estrutura de dados HyP_data.
    # Por exemplo, remover o elemento ('feature') que está no índice 2.
    feature_index = 0  # Índice do elemento a remover.  Feature 'test_only'
    dataset_HyP['Features'].pop(feature_index)

    #feature_index = 3  # Índice do elemento a remover.  Feature 'Band-Energy-Ratio'
    #dataset_HyP['Features'].pop(feature_index)


    # # Utilizar apenas uma 'feature'
    # name_of_the_feature = "Band-Energy-Ratio"  # feature_name
    # features_to_use = []
    # for i, feature in enumerate(dataset_HyP['Features']):  # 'i' -> index
    #
    #     if feature['feature_name'] == name_of_the_feature:
    #         features_to_use.append(feature)
    #
    # dataset_HyP['Features'] = features_to_use


    # Criar o HyP_data_Dataset_xxx.json com a estrutura de dados definida para o Dataset_xxx
    HyP_data_Dataset_xxx = path_to_root + HyP_data_DATASET_PATH + '/HyP_data_Dataset_N_'+str(n)+'.json'

    # Opening JSON file in 'write' mode
    with open(HyP_data_Dataset_xxx, 'w') as HyP_data_Dataset_xxx_file:
        json.dump(dataset_HyP, HyP_data_Dataset_xxx_file, indent=4)

    HyP_data_Dataset_xxx_file.close()


    CsvFile.remove_file(path_to_root + dataset_XXX_file)
    features_file = CsvFile(path_to_root + dataset_XXX_file, "w")

    file_rows.clear()

    construct_dataset(paths_train, dataset_HyP, features_file, file_rows)

