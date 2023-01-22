# ################################################################################################################################################
# Edit "parent_folder" to show the parent folder of the classification results folder
# The folder is supposed to have a structure like this:
"""
├── Ens.Resultsroc_aucAdaboost
│   ├── All_ET_Combinations123456_4Python_ConfMatrix.pdf
│   ├── All_ET_Combinations123456_4Python_ROC.pdf
│   ├── All_ET_Combinations123456_4Python_gridsearch_object.model
│   ├── All_ET_Combinations123456_4Pythonclassification_report.csv
│   ├── All_ET_Combinations12345_4Python_ConfMatrix.pdf
│   ├── All_ET_Combinations12345_4Python_ROC.pdf
│   ├── All_ET_Combinations12345_4Python_gridsearch_object.model
│   ├── All_ET_Combinations12345_4Pythonclassification_report.csv
│   ├── All_ET_Combinations12346_4Python_ConfMatrix.pdf
│   ├── All_ET_Combinations12346_4Python_ROC.pdf
│   ├── All_ET_Combinations12346_4Python_gridsearch_object.model
│   ├── All_ET_Combinations12346_4Pythonclassification_report.csv
│   ├── All_ET_Combinations1234_4Python_ConfMatrix.pdf
│   ├── All_ET_Combinations1234_4Python_ROC.pdf
│   ├── All_ET_Combinations1234_4Python_gridsearch_object.model
│   ├── All_ET_Combinations1234_4Pythonclassification_report.csv
│   ├── All_ET_Combinations12356_4Python_ConfMatrix.pdf
...
├── Ens.Resultsroc_aucExtraTreesClassifier
│   ├── All_ET_Combinations123456_4Python_ConfMatrix.pdf
│   ├── All_ET_Combinations123456_4Python_ROC.pdf
│   ├── All_ET_Combinations123456_4Python_gridsearch_object.model
│   ├── All_ET_Combinations123456_4Pythonclassification_report.csv
│   ├── All_ET_Combinations12345_4Python_ConfMatrix.pdf
│   ├── All_ET_Combinations12345_4Python_ROC.pdf
│   ├── All_ET_Combinations12345_4Python_gridsearch_object.model
│   ├── All_ET_Combinations12345_4Pythonclassification_report.csv
│   ├── All_ET_Combinations12346_4Python_ConfMatrix.pdf
│   ├── All_ET_Combinations12346_4Python_ROC.pdf
│   ├── All_ET_Combinations12346_4Python_gridsearch_object.model
│   ├── All_ET_Combinations12346_4Pythonclassification_report.csv
│   ├── All_ET_Combinations1234_4Python_ConfMatrix.pdf
│   ├── All_ET_Combinations1234_4Python_ROC.pdf
│   ├── All_ET_Combinations1234_4Python_gridsearch_object.model
│   ├── All_ET_Combinations1234_4Pythonclassification_report.csv
│   ├── All_ET_Combinations12356_4Python_ConfMatrix.pdf
│   ├── All_ET_Combinations12356_4Python_ROC.pdf
│   ├── All_ET_Combinations12356_4Python_gridsearch_object.model
│   ├── All_ET_Combinations12356_4Pythonclassification_report.csv
│   ├── All_ET_Combinations1235_4Python_ConfMatrix.pdf
│   ├── All_ET_Combinations1235_4Python_ROC.pdf
│   ├── All_ET_Combinations1235_4Python_gridsearch_object.model
│   ├── All_ET_Combinations1235_4Pythonclassification_report.csv
...
...
"""
# each folder contains at several files of type "*.model" that are the best fitted models through a gridcv search
# This script also to set "folder_path_data_splits" to point to the directory containing the datasplit objects of type "*.dsplt"
## There should be a 1-to-1 mapping between "*.dsplt" and "*.model" files.
# It parses each folder and generates two detailed performance details files in "PerfDetails" sub-folder for each classification results
# ################################################################################################################################################
# NOTES:
# 1- don't forget to put "/" at the end of the paths
# 2- set the sklearn version to 1.0.2 otherwise you'll get error:
#   you can uninstall the current sklearn version by running "pip uninstall scikit-learn"
#   and install the specified version by running this: "pip install scikit-learn==1.0.2"
#   you can check the sklearn version by running the below commands in the python console:
#   import sklearn
#   sklearn.__version__
# ################################################################################################################################################

#the two first lines will be used for running on a server (avoiding the no display name error )
import matplotlib
#matplotlib.use('Agg')
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import glob
import numpy as np
from collections import namedtuple
import pickle
import sklearn.neighbors._base
import sys
import shutil






# # Function def

def detailed_perf_row(fitted_model, x_matrix , y_true):
    """
    # Description:
    This function calculates detailed performance measures
    ("N: Total","N: ASD","N: nonASD","tp","fp","tn","fn","SN","SP","Pre aka PPV", "NPV", "ACC", "F1-Score", "AUC ROC","Model Details")
    on a given dataset.
    # Input:
    "fitted_model": trained model object for a ET combination
    "x_matrix": feature matrix
    "y_true": corresponding labels for the "x_matrix"

    # Output:
    "perf_parameters_list" a list of performance measure values along with some details:
    "N: Total","N: ASD","N: nonASD","tp","fp","tn","fn")
     """
    #applying the fitted model on the data
    y_pred = fitted_model.predict(x_matrix)
    #calculating confustion matrix
    conf_mat = confusion_matrix(y_true, y_pred, labels=["ASD", "nonASD"])
    #extracting the four basic paramteres
    tp = conf_mat[0,0]
    fn = conf_mat[0,1]
    fp = conf_mat[1,0]
    tn = conf_mat[1,1]
    sample_size = np.sum(conf_mat)
    #calculating preformance measures
    sen = tp / (tp + fn)
    spc = tn / (tn + fp)
    acc = (tp + tn)/(tp + tn + fp + fn)
    pre = tp / (tp + fp)#aka PPV
    npv = tn / (tn + fn)
    f1_score = 2 * (pre * sen)/(pre + sen)
    auc_roc = roc_auc_score(y_true, fitted_model.predict_proba(x_matrix)[:, 1])
    #list of all performance measures
    #["N: Total","N: ASD","N: nonASD","tp","fp","tn","fn","SN","SP","Pre aka PPV", "NPV", "ACC", "F1-Score", "AUC ROC"]
    perf_parameters_list = [sample_size,tp+fn,tn+fp,tp,fp,tn,fn,sen,spc,pre,npv,acc,f1_score, auc_roc, str(fitted_model.best_params_)]
    return perf_parameters_list


def read_et_dataset_and_train_test_split(full_file_path):

    et_data = pd.read_csv(full_file_path, index_col=0)  # index_col=0 ==> has row name
    et_data = pd.DataFrame(et_data)
    # removing the non-Complete cases (those cases that includes NA)
    et_data.dropna(axis=0, how='any', inplace=True)
    # extracting the feature matrix and label vector
    X = et_data.iloc[:, :-1]
    y = et_data.iloc[:, -1]
    #using random_state makes it reproducible
    return train_test_split(X, y, test_size=0.2, random_state=6156)







# # Reading all data set file names




def load_and_list_model_and_data_split_objects(folder_path_models, folder_path_data_splits):

    """
    # Description:
    This function reads all fitted models and data splits where are in the "folder_path_models", "folder_path_data_splits" folders, respectively.
    and selecting the optimal classifier and writing the validation results for the given file.
    This file makes train/validation/test split, and uses the train for fitting the models and validation for the assessing the fitted model.
    Finally, saves the best model, according to the grid search result.
    # Input:
    "folder_path_models": a string which is the full path for the folder containing trained model objects (extention of those files are ."model")
    "folder_path_data_splits": a string which is the full path for the folder containing data splits objects (extention of those files are ."dsplt")
    # Output:
    a tuple of two lists each containig the full file name of those objects: (all_trained_model_objects_file_name, all_data_split_objects)
    this tuple will be used to load the models and apply them on the split data.
    """
    # listing all trained models object's full path
    folder_path_models += "*.model"
    all_trained_model_objects_file_name = glob.glob(folder_path_models)
    # sorting the list to be matched with the data split objects
    all_trained_model_objects_file_name.sort()
    # listing all data split object's full path
    folder_path_data_splits += "*.dsplt"
    all_data_split_objects = glob.glob(folder_path_data_splits)
    # sorting the list to be matched with the trained models objects
    all_data_split_objects.sort()
    return (all_trained_model_objects_file_name, all_data_split_objects)


def calculate_detailed_perf_df(data_splt_model_tuple):
    """
    # Description:
    This function calculates detailed performance measures
    ("N: Total","N: ASD","N: nonASD","tp","fp","tn","fn","SN","SP","Pre aka PPV", "NPV", "ACC", "F1-Score", "AUC ROC","Model Details")
    on train and validation sets for all ET combinations.
    # Input:
    "data_splt_model_tuple": a list of all:a trained model; and the path of a data split object corresponds to the models for all ET combinations

    # Output:
    "perf_detailed_df_total" a detailed dataframe of performance measures on all ET combination
    by calling "detailed_Perf_train_validation" for each ET combinaion.
    """

    #init the final performance measure dataframe 
    perf_measures_whole_df = pd.DataFrame(columns=["ET Combination","Train/Validation",
                          "N: Total","N: ASD","N: nonASD","tp","fp","tn","fn","SN","SP",
                          "Pre aka PPV", "NPV", "ACC", "F1-Score", "AUC ROC","Model Details"])
    perf_detailed_df_list = []
    for trainde_model, data_split in zip(data_splt_model_tuple[0], data_splt_model_tuple[1]):
        #extracting the name of the ET 
        et_combination_name = data_split.split("_4Python")[0]
        et_combination_name = et_combination_name.split("Combinations")[1]
        #print(et_combination_name)
        #loading the trained (fitted) model object
        current_fitted_model = joblib.load(trainde_model)
        #loading data split object
        data_split_obj = joblib.load(data_split)# the structure of the data split object: dict(zip(['X_train', 'X_validation', 'X_test', 'y_train', 'y_validation', 'y_test'], [X_train, X_validation, X_test, y_train, y_validation, y_test]))
        current_perf_detailed_df = detailed_Perf_train_validation(current_fitted_model, data_split_obj, et_combination_name)
        perf_detailed_df_list.append(current_perf_detailed_df)
    perf_detailed_df_total = pd.concat(perf_detailed_df_list)    
    return perf_detailed_df_total
    

    

def detailed_Perf_train_validation(fitted_model, data_split, et_combination_name):
    """
    # Description:
    This function calculates detailed performance measures
    ("N: Total","N: ASD","N: nonASD","tp","fp","tn","fn","SN","SP","Pre aka PPV", "NPV", "ACC", "F1-Score", "AUC ROC","Model Details")
    on train and validation sets for a specific ET combination.
    # Input:
    "fitted_model": trained model object for a ET combination
    "data_split": data split object for the ET combination
    "et_combination_name": name of the ET combination

    # Output:
    "perf_measures_df" a dataframe with two rows of performance measures on the ET combination
    by calling "detailed_perf_row" on the train and validation data.
    """

    #calculating the preformance measures for training data
    perf_details_train = detailed_perf_row(fitted_model, data_split['X_train'], data_split['y_train'])
    #adding name to the results
    perf_details_train = ["Train"] + perf_details_train
    #adding ET Combination
    perf_details_train = [et_combination_name] + perf_details_train

    #calculating the preformance measures for validation data
    perf_details_validation = detailed_perf_row(fitted_model, data_split['X_validation'], data_split['y_validation'])
    #adding name to the results
    perf_details_validation = ["Validation"] + perf_details_validation
    #adding ET Combination
    perf_details_validation = [et_combination_name] + perf_details_validation
    perf_measures_df =    pd.DataFrame(columns=["ET Combination","Train/Validation",
                      "N: Total","N: ASD","N: nonASD","tp","fp","tn","fn","SN","SP",
                      "Pre aka PPV", "NPV", "ACC", "F1-Score", "AUC ROC", "Model Details"])
    perf_measures_df.loc[et_combination_name + ": Train"] = perf_details_train
    perf_measures_df.loc[et_combination_name + ": Validation"] = perf_details_validation
    return perf_measures_df


    








def write_summary_results(full_path_4_results_df):
    """
    # Description:
    This function writes a summary result dataframe according to Karens's suggestions.
    # Input:
    "full_path_4_results_df": path to the detailed data frame, wich was generated by "driver_fun_extracting_detailed_perf_measures"

    # Output:
    Not explicit. It writes the detailed data frame as a csv file ("summary_file_full_name.csv") in full_path_4_results_df folder.
    """
    # Karen Preferred format
    # NOTE: this function assumes that the csv file has a row name column (the first col)
    #read the csv file generated by "driver_fun_extracting_detailed_perf_measures"
    print("HORAY\n\n\n\n\n\n\n\n\n\n\n\n")
    detailed_per_df = pd.read_csv(full_path_4_results_df)
    
    #generating name to be used as the row name
    name = list(range(detailed_per_df.shape[0]))
    #set_trace()
    #adding as a new column to the df
    detailed_per_df['name'] = name
    
    #assigning as row name 
    detailed_per_df.set_index('name')
    
    #filtering: selecting only validation data
    detailed_per_df_validation_only = detailed_per_df.loc[detailed_per_df["Train/Validation"].isin(["Validation"])]
    
    #selecting the columns that are correspond to performance measures (these columns are numerical and so we can apply "idmax" method on the resulting df)
    detailed_per_df_validation_only_perf_columns = detailed_per_df_validation_only[[ 'SN', 'SP', 'Pre aka PPV', 'NPV', 'ACC','F1-Score', 'AUC ROC',]]
    
    #finding the row indeces that corresponds to the maximum values in each columns
    row_indx_for_max_values = detailed_per_df_validation_only_perf_columns.idxmax(axis=0)
    
    #subsetting the "detailed_per_df_validation_only" df to have only maximum-perf-associated rows (argmax)
    final_selected_rows = detailed_per_df_validation_only.loc[row_indx_for_max_values,:]
    
    #adding a new column to have the name of the best per measure which is correspond to each row
    final_selected_rows["OptimizedMeasure"]= detailed_per_df_validation_only_perf_columns.idxmax(axis=0).index
    
    #moving the last column to the first
    col_name_list = list(final_selected_rows.columns.values)
    reordered_col_names = [col_name_list[-1]] + col_name_list[:-1]    
    final_selected_rows = final_selected_rows[reordered_col_names]
    
    #preparing the full file name
    full_path_4_results_df_splitted_by_slash = full_path_4_results_df.split("/")
    summary_file_folder_path = "/".join(full_path_4_results_df_splitted_by_slash[:-1]) 
    summary_file_full_name =  summary_file_folder_path + "/DetailedPerfDataFrameSummary.csv"

    #writing as a csv file
    final_selected_rows.to_csv(summary_file_full_name)




def driver_fun_extracting_detailed_perf_measures(folder_path_models, folder_path_data_splits):
    """
    # Description:
    Driver function for calculating detailed perf. measures.
    # Input:
    "folder_path_models": path to the trained model objects
    "folder_path_data_splits": path to the data split objects

    # Output:
    Not explicit. It writes the detailed data frame as a csv file ("DetailedPerfDataFrame.csv") in folder_path_models/"PerfDetails" subfolder (which will be created in this fuction).
    """
    #making directory for writing the detailed performance results
    #if the directory exist remove it 
    dir_path = folder_path_models + "PerfDetails"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)    
    os.mkdir(dir_path)
    #folder_path_data_splits = "/Users/apple/Desktop/ET_Combination_4_jabba/ARI/ET_Combination_4_jabba/DataSplits/"
    data_splt_model_tuple = load_and_list_model_and_data_split_objects(folder_path_models, folder_path_data_splits)
    
    detailed_perf_df = calculate_detailed_perf_df(data_splt_model_tuple)
    full_path_4_results_df = dir_path+"/DetailedPerfDataFrame.csv"
    #adding row name to the "detailed_perf_df" 
    #pass
    #writing the final df as a csv file
    detailed_perf_df.to_csv(full_path_4_results_df)    
    #writing the summary result file
    write_summary_results(full_path_4_results_df)    


# ![PythonProject-2.jpg](attachment:PythonProject-2.jpg)

# # Calling the driver function

# In[8]:



parent_folder = "/Users/Javad/Desktop/Karen/ARICleanRes/All_ET_DataCombination/aucResults/Debugging/"
results_folder_list = [x for x in os.walk(parent_folder)][0][1]
folder_path_models_list = [parent_folder+folder_name for folder_name in results_folder_list]
folder_path_data_splits = "/Users/Javad/Desktop/Karen/ARICleanRes/DataSplits/" #!! put the ending '/'
for folder_path_models in folder_path_models_list:
    print(folder_path_models)
    print("==================================================\n\n")
    try:
        driver_fun_extracting_detailed_perf_measures(folder_path_models + "/", folder_path_data_splits)
    except:
        print("Oops!", sys.exc_info()[0], "occurred.")
        print("Next entry.")
        print(folder_path_models)
            



