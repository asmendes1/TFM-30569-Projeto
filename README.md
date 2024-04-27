# Hyperparameter Optimization for Impulsive Sound Classifiers
This is an academic final project, Master's Degree in Informatics and Computer Engineering.

ISEL – Instituto Superior de Engenharia de Lisboa, Rua Conselheiro Emídio Navarro, 1959-007 Lisbon, Portugal


## Authors

	André Mendes
	
	Prof. Doutor Paulo Trigo (as advisor)
	Prof. Doutor Joel P. Paulo (as advisor)

 	Affiliation: ISEL – Instituto Superior de Engenharia de Lisboa, Rua Conselheiro Emídio Navarro, 1959-007 Lisbon, Portugal

April, 2024


# 

1. Software version

Python version used in the project development: version 3.11


# 

2. Structure of the project: directories (D) and files (F) 

(D) Configurations

	(F) constants.py
	Initial constants for the project. (Help avoiding the hardcoded data)
 	The ..._PATH constants represent the path from the project root directory.


(D) Data

	(D) videos
 	The videos to be processed by the system are placed in this directory. The file name of each video must end with « _<video order number> ».
  
  	(D) audios
   	In this directory, the system stores the audio extracted from the videos.
    
    	(D) labeling 
     	In this directory, files related to the human (ear) annotation of sound events are placed. The name of each file must be « labeling_<order number of the corresponding video> ». Each line of the file must be composed of: <type of event>;<court side>;<first sample>;<last sample>.

      
(D) Hyperparameters

	(F) HyP_data_scheme.json
 
 	(F) HyP_learn_scheme.json
  	Data structure that formally represents the HyP_data and HyP_learn scheme. The structure must be preserved, so that the information can be processed in a generic way by the system.


(D) my_utils

	(F) classifier_fuctions.py
 	Module that contains the functions to get the classifiers. This file can be used to extend the system by incorporating additional functions (plug-in model) to explore other classifiers.
  
  	(F) feature_extraction_fuctions.py
   	Module that contains the functions to extract audio features. Each function receives the following parameters: the audio file as a floating point time series; all HyP_data_global ; and the specific parameters to the respective audio feature that have been defined in the HyP_data schema. This file can be used to extend the system by incorporating additional functions (plug-in model) to explore other audio features.

	(F) processCsvFile.py
 	Utility functions for processing CSV files.

	(F) videoProcessing.py
 	Utility functions for processing videos.


(D) Project

	(D) DatasetCreation
 
 		(F) createDataset.py
   		This is the main file for the dataset creation. 
     
     		(F) datasetCreator.py
       		Auxiliary file for the dataset creation.
	 
  	(D) DatasetStorage
   	In this directory, the system stores the datasets created, as well as the HyP_data of each one.
    
    (D) Model
     
     	(F) Pipeline_run_Grid_search.py    
       
       	(F) Pipeline_run_Bayes_search.py
	 		Both files are the main for the datasets processing, and can be executed independently. They perform the same task, but one of them uses the grid search technique, and the other uses the Bayes optimization technique.


(D) Results

In this directory, the system stores the final results in a CSV file. (The final results are also printed in the console output)


# 

3. To run the project

Make sure you put the videos in the Data\videos folder, and the labeling files in the Data\labeling folder. Check the HyP_data and HyP_learn schemas.

1st phase – execute the file createDataset.py

2nd phase (only after the 1st phase is completed) – execute the file Pipeline_run_Grid_search.py and/or the file Pipeline_run_Bayes_search.py


