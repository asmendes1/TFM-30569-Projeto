
import os
import json
import copy
import pandas as pd
from datetime import datetime
from numpy import mean
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import time
from Configurations.constants import *
import my_utils.classifier_functions as cf



def converter_params_para_python(params):
    params_python = {}

    for param, config in params.items():
        param_type = config['param_type']

        if param_type == 'tuple':
            params_python[param] = config['values']
        elif param_type == 'categorical':
            params_python[param] = config['values']
        elif param_type == 'real':
            range_info = config['range']
            min_value = range_info['min']
            max_value = range_info['max']
            step = range_info['step']
            params_python[param] = np.arange(min_value, max_value, step)
        elif param_type == 'int':
            range_info = config['range']
            min_value = range_info['min']
            max_value = range_info['max']
            step = range_info['step']
            params_python[param] = np.arange(min_value, max_value, step, dtype=int)

    return params_python




# Configuration to display all DataFrame columns without ellipsis, and also to not truncate the values
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# Current file path relative to the project root.
path_to_root = "../../"

dataset_storage_path = path_to_root + DATASET_STORAGE_PATH
hyp_learn_scheme_file_path = path_to_root + HYP_LEARN_SCHEME_FILE_PATH
results_path = path_to_root + RESULTS_PATH
performance_metric = PERFORMANCE_METRIC
folds_outer = NESTED_CROSS_VALIDATION_FOLDS_OUTER
folds_inner = NESTED_CROSS_VALIDATION_FOLDS_INNER



# Checks if the directory exists
if os.path.exists(dataset_storage_path):
    # List all files in the directory
    all_files = os.listdir(dataset_storage_path)

    # Filter only files with .csv extension
    files_csv_only = [arquivo for arquivo in all_files if arquivo.endswith('.csv')]

    scores = []
    inicio = time.time()

    for dataset_filename in files_csv_only:
        dataset_filepath = os.path.join(dataset_storage_path, dataset_filename)

        df = pd.read_csv(dataset_filepath)

        # Defining the features and the target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]


        # Opening JSON file in 'read' mode
        with open(hyp_learn_scheme_file_path, 'r') as HyP_learn_file:
            # returns JSON object as a dictionary
            HyP_learn_scheme = json.load(HyP_learn_file)

        HyP_learn_file.close()

        skf_cv_outer = StratifiedKFold(n_splits=folds_outer, shuffle=True, random_state=1)
        skf_cv_inner = StratifiedKFold(n_splits=folds_inner, shuffle=True, random_state=1)

        for algorithm_data in HyP_learn_scheme["algorithms"]:
            algorithm_name = algorithm_data["name"]
            algorithm_params_json = algorithm_data["params"]

            print(f"Training and testing {algorithm_name}...")

            funcao = getattr(cf, algorithm_data['function_name'])
            params = algorithm_params_json

            classifier_algorithm = funcao(params)

            # Convert to Python representation
            params_python = converter_params_para_python(algorithm_params_json)

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', classifier_algorithm)
            ])

            performance_results = list()
            best_performance = 0
            best_HyP_learn = {}
            for train_idx, test_idx in skf_cv_outer.split(X, y):
                X_train = X.iloc[train_idx, :]
                y_train = y.iloc[train_idx]

                X_test = X.iloc[test_idx, :]
                y_test = y.iloc[test_idx]

                if performance_metric == "accuracy_score":
                    search = GridSearchCV(estimator=pipeline, param_grid=params_python, scoring='accuracy',
                                          cv=skf_cv_inner, refit=True)

                    inner_result = search.fit(X_train, y_train)
                    best_model_inner = inner_result.best_estimator_

                    model_inner_predict = best_model_inner.predict(X_test)
                    performance_best_model_inner = accuracy_score(y_test, model_inner_predict)

                    performance_results.append(performance_best_model_inner)

                    if performance_best_model_inner > best_performance:
                        best_performance = performance_best_model_inner
                        best_HyP_learn = copy.deepcopy(inner_result.best_params_)

                elif performance_metric == "f1_score":
                    search = GridSearchCV(estimator=pipeline, param_grid=params_python, scoring='f1',
                                          cv=skf_cv_inner, refit=True)

                    inner_result = search.fit(X_train, y_train)
                    best_model_inner = inner_result.best_estimator_

                    model_inner_predict = best_model_inner.predict(X_test)
                    performance_best_model_inner = f1_score(y_test, model_inner_predict)

                    performance_results.append(performance_best_model_inner)

                    if performance_best_model_inner > best_performance:
                        best_performance = performance_best_model_inner
                        best_HyP_learn = copy.deepcopy(inner_result.best_params_)

                else:
                    print("The performance metric could not be identified.")

            scores.append({
                'dataset': dataset_filename,
                'model': algorithm_name,
                'mean_score': mean(performance_results),
                'best_params': copy.deepcopy(best_HyP_learn)
            })

    fim = time.time()
    tempo_decorrido = fim - inicio
    horas = int(tempo_decorrido // 3600)
    tempo_decorrido %= 3600
    minutos = int(tempo_decorrido // 60)
    segundos = int(tempo_decorrido % 60)

    # DataFrame
    df = pd.DataFrame(scores, columns=['dataset', 'model', 'mean_score', 'best_params'])

    # Ordenar o DataFrame em função da coluna 'best_score'
    df_ = df.sort_values(by='mean_score', ascending=False)
    print('========== FINAL RESULTS (GridSearch) ==========')
    print(df_)

    # print(f"Tempo decorrido: {tempo_decorrido:.2f} segundos")
    print(f"Tempo decorrido: {horas} horas, {minutos} minutos e {segundos} segundos")
    print('---------------------------------------')

    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    filename = f'results_GridSearch_{current_datetime}.csv'

    # Save results to a CSV file
    df_.to_csv(results_path + "/" + filename, index=True)

else:
    print("The specified directory does not exist.")

