
import matplotlib.pyplot as plt
import copy
import pandas as pd
from numpy import mean
import numpy as np
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC     # Support Vector Classifier
import time

# Configuração para exibir todas as colunas do DataFrame sem reticências, e também, para não truncar os valores
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)  # Definir o máximo de largura de coluna para não truncar os valores

DATASET_STORAGE_PATH = "Project/DatasetStorage"

# Caminho do ficheiro atual relativamente à raiz do projecto. (se eu colocar o presente ficheiro noutra directoria, basta verificar esta variável)
path_to_root = "../../"




#dataset_filename = path_to_root + DATASET_STORAGE_PATH + '/features_file_Carina.csv'
dataset_filename = '/Dataset_N_5_test_only.csv'
dataset_filename_and_path = path_to_root + DATASET_STORAGE_PATH + '/Dataset_N_5_test_only.csv'

# Carregando o dataset
df = pd.read_csv(dataset_filename_and_path)

# Definindo as features e o target
X = df.iloc[:, :-1] # X = df.drop('target', axis=1)
y = df.iloc[:, -1]  # y = df['target']


# print('---------------------- y ------------------')
# print(y)
#
# print('---------------------- X ------------------')
# print(X)


# # Dividindo o dataset em conjunto de treinamento e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============== um classificador MLP e um classificador SVM em pipelines separados ================
# Criando um pipeline com a Rede Neural (MLP)
pipeline_mlp = Pipeline([
    ('scaler', StandardScaler()),        # Etapa de pré-processamento: Normalização dos dados
    ('mlp', MLPClassifier())             # Etapa do classificador: Rede Neural MLP
])

# Criando um pipeline com a Support Vector Machine
pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),      # Etapa de pré-processamento: Normalização dos dados
    ('svm', SVC())       # Etapa do classificador: SVM
])


# Definindo os hiperparâmetros (que queremos otimizar) e seus respectivos espaços de pesquisa
param_grid_GridSearch = {
    'mlp':{
        'model': pipeline_mlp,
        'params': {
            'mlp__learning_rate_init': np.arange(0.0001,0.01,0.0005),  #[0.0005, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.01], # Espaço de pesquisa para a taxa de aprendizagem
            # 'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)], # numero de camadas da rede; número de neurónios em cada camada
            # 'activation': 'relu',          # Função de ativação
            # 'solver': 'adam',              # Otimizador
            # 'alpha': 0.0001,               # Regularização L2
            # 'batch_size': 'auto',          # Tamanho do lote
            'mlp__max_iter': np.arange(1,100,5)
        }
    },
    'svm': {
        'model': pipeline_svm,
        'params': {
            #'svm__kernel': ['linear','poly'],
            #'svm__gamma': 'scale',     # Parâmetro gamma para kernels RBF
            'svm__C': np.arange(1,40,3)   # Parâmetro de regularização

        }
    }
}

skf_cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
skf_cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

scores = []
total_combinations_grid = 0

inicio = time.time()

for model_name, mp in param_grid_GridSearch.items():

    accuracy_results = list()
    best_accuracy = 0
    best_HyP_learn = {}
    for train_idx, test_idx in skf_cv_outer.split(X, y):
        X_train = X.iloc[train_idx, :]
        y_train = y.iloc[train_idx]

        X_test = X.iloc[test_idx, :]
        y_test = y.iloc[test_idx]


        #clf = GridSearchCV(estimator=mp['model'], param_grid= mp['params'], scoring='accuracy', cv=5, return_train_score=False)

        search = GridSearchCV(estimator=mp['model'], param_grid= mp['params'], scoring='accuracy', cv=skf_cv_inner, refit=True)

        inner_result = search.fit(X_train, y_train)
        best_model_inner = inner_result.best_estimator_

        # número total de combinações testadas pelo GridSearchCV
        total_combinations_grid += len(inner_result.cv_results_['params'])

        model_inner_predict = best_model_inner.predict(X_test)
        accuracy_best_model_inner = accuracy_score(y_test, model_inner_predict)

        accuracy_results.append(accuracy_best_model_inner)

        if accuracy_best_model_inner > best_accuracy:
            best_accuracy = accuracy_best_model_inner
            best_HyP_learn = copy.deepcopy(inner_result.best_params_)

        # execute the nested cross-validation
        # search_outer = cross_val_score(estimator=search, X=X, y=y, scoring='accuracy', cv=skf_cv_outer, n_jobs=-1)

    scores.append({
        'dataset': dataset_filename,
        'model': model_name,
        'mean_score': mean(accuracy_results), # 'Accuracy' média dos melhores modelos de cada fold da camada externa
        'best_params': copy.deepcopy(best_HyP_learn) # melhor conjunto de HyP_learn de entre os melhores de cada fold da camada externa. Não pode ser a média dos melhores, porque não dá para calcular a média, visto que há parâmetros que são literais e não numéricos. (Isto poderá ser aceite assim????)
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
print('========== RESULTADOS FINAIS ==========')
print(df_)
# Total de combinações de parâmetros testadas
print("Número total de combinações testadas pelo GridSearchCV:", total_combinations_grid)
#print(f"Tempo decorrido: {tempo_decorrido:.2f} segundos")
print(f"Tempo decorrido: {horas} horas, {minutos} minutos e {segundos} segundos")
print('---------------------------------------')
