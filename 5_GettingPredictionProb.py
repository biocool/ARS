#!/usr/bin/env python
# coding: utf-8

# In[5]:


#imports

#the two first lines will be used for running on a server (avoiding the no display name error )
import matplotlib
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
import pickle
import sklearn.neighbors._base
import sys
import shutil
from IPython.core.debugger import set_trace


# In[ ]:





# In[10]:


################################################################################################################################
# # Reading all data set file names
def load_and_list_model_and_data_split_objects(folder_path_models, folder_path_data_splits):
    
    '''
    # Description:
    This function loads all fitted models and data splits where are in the "folder_path_models" and "folder_path_data_splits" folders, respectively.    

    Args:  
    param0:"folder_path_models": a string which is the full path for the folder contianing trained model objects (extention of those files are ."model")
    param1:"folder_path_data_splits": a string which is the full path for the folder contianing data splits objects (extention of those files are ."dsplt")
    
    Returns:
    a tuple of two lists each containig the full file name of those objects: (all_trained_model_objects_file_name, all_data_split_objects)
    this tuple will be used to load the models and apply them on the split data.
    '''    
    
    
    #set_trace()
    #listing all trained models object's full path
    folder_path_models += "*.model"
    all_trained_model_objects_file_name = glob.glob(folder_path_models)

    #sorting the list to be matched with the data split objects
    all_trained_model_objects_file_name.sort()

    #listing all data split object's full path
    folder_path_data_splits += "*.dsplt"
    all_data_split_objects = glob.glob(folder_path_data_splits)
    
    #sorting the list to be matched with the trained models objects
    all_data_split_objects.sort()
    return (all_trained_model_objects_file_name, all_data_split_objects)



################################################################################################################################
def calculate_detailed_pred_ARS(data_splt_model_tuple, results_dir_path):

    '''
    # Description:
    This function calls "detailed_prediction_ARS_train_validation" on all (trainde_model, data_split) pairs to calculate the ARS for each ET_combination.
    
    Args:
    "data_splt_model_tuple": a list of all:a trained model; and the path of a data split object corresponds to the models for all ET combinations
    "results_dir_path": a directory path for writing the results (it will be passed to the "detailed_prediction_ARS_train_validation" function)

    Returns:
    Void
    '''

    
    #set_trace()
    #looping over all trained model and corresponding data split objects to apply the models on the train/validation sets
    #data_splt_model_tuple[0] is the list of trained model objects and data_splt_model_tuple[1] is the list of data split obejcts
    
    for trainde_model, data_split in zip(data_splt_model_tuple[0], data_splt_model_tuple[1]):

        #extracting the name of the ET 
        et_combination_name = data_split.split("_4Python")[0]
        et_combination_name = et_combination_name.split("Combinations")[1]

        #loading the trained (fitted) model object
        current_fitted_model = joblib.load(trainde_model)

        #loading data split object
        data_split_obj = joblib.load(data_split)# the structure of the data split object: dict(zip(['X_train', 'X_validation', 'X_test', 'y_train', 'y_validation', 'y_test'], [X_train, X_validation, X_test, y_train, y_validation, y_test]))
        detailed_prediction_ARS_train_validation(current_fitted_model, data_split_obj, et_combination_name, results_dir_path)




################################################################################################################################    
def detailed_prediction_ARS_train_validation(fitted_model, data_split, et_combination_name, results_dir_path):
    
    '''
    Description:
    This function calls "detailed_prediction_res_with_ARS" on two (trainde_model, data_split) pairs (for train and validation datasets) to calculate the ARS.
    
    Args:
    "fitted_model": trained model object for a ET combination
    "data_split": data split object for the ET combination
    "et_combination_name": name of the ET combination
    "results_dir_path": a directory path for writing the results (it will be passed to the "detailed_prediction_res_with_ARS" function)

    Returns:
    Not explicit. It wirtes the ARS details results as two csv files (for train and test datasets)
    
    '''    


    #####calculating and writing ARS for the training dataset#####
    #calculating the ars for training data
    ars_details_train = detailed_prediction_res_with_ARS(fitted_model, data_split['X_train'], data_split['y_train'])    
    #full path for writng the results
    ars_train_file_full_path = results_dir_path + et_combination_name + "_ARS_Train.csv"     
    #writing as a csv file
    ars_details_train.to_csv(ars_train_file_full_path)
    
    #####calculating and writing ARS for the validation dataset#####
    #calculating the ars for the validation data
    ars_details_validation = detailed_prediction_res_with_ARS(fitted_model, data_split['X_validation'], data_split['y_validation'])    
    #full path for writng the results
    ars_validation_file_full_path = results_dir_path + et_combination_name + "_ARS_Validation.csv"     
    #writing as a csv file
    ars_details_validation.to_csv(ars_validation_file_full_path)







################################################################################################################################
def detailed_prediction_res_with_ARS(fitted_model, x_matrix , y_true):
    
    '''
    # Description:
    This function calls "detailed_prediction_res_with_ARS" on two (trainde_model, data_split) pairs (for train and validation datasets) to calculate the ARS for each ET_combination.
    
    Args:
    "fitted_model": trained model object for a ET combination
    "x_matrix": feature matrix
    "y_true": corresponding labels for the "x_matrix"

    Returns:
    "final_detailed_df_4_ARS": a pd.df which is column-wise concatenation of the four dfs [x_matrix, y_true_df, y_pred_df, prediction_prob_ASD_df]
    '''    


    
    #applying the fitted model on the data
    y_pred = fitted_model.predict(x_matrix)
    #converting to pd.df
    y_pred_df = pd.DataFrame(y_pred)    
    #assigning colname
    y_pred_df.columns = ['Predicted_Class']    
    #assigning the same row namse as x_matrix (which is the subject IDs)
    y_pred_df.index = x_matrix.index
    
    

    #converting to pd.df
    y_true_df = pd.DataFrame(y_true)    
    #assigning colname
    y_true_df.columns = ['Djx']    
    #assigning the same row namse as x_matrix (which is the subject IDs)
    y_true_df.index = x_matrix.index

    #extracting the prediction probabilities 
    prediction_prob_nd_arr = fitted_model.predict_proba(x_matrix)    
    #extracting the prediction probabilities for ASD class
    prediction_prob_ASD = prediction_prob_nd_arr[:, 0]    
    #converting to pd.df
    prediction_prob_ASD_df = pd.DataFrame(prediction_prob_ASD)    
    #assigning colname
    prediction_prob_ASD_df.columns = ['ARS_first_attempt']    
    #assigning the same row namse as x_matrix (which is the subject IDs)
    prediction_prob_ASD_df.index = x_matrix.index
    
    #####final data frame#####
    
    #concatenating all dfs    
    #axis=1 ==> column wise concatenation 
    final_detailed_df_4_ARS = pd.concat([x_matrix, y_true_df, y_pred_df, prediction_prob_ASD_df], axis=1) 
    
    

    return final_detailed_df_4_ARS





################################################################################################################################
def driver_fun_extracting_ARS(folder_path_models, folder_path_data_splits):
    '''
    # Description:
    Driver function for calculating detailed ARS. 
    
    Args:
    "folder_path_models": path to the trainde model objects
    "folder_path_data_splits": path to the data split objects

    Returns:
    Not explicit. 
    '''


    #making directory for writing the detailed ARS results
    #if the directory exist remove it 
    ars_dir_path = folder_path_models + "ARS_Res"
    if os.path.exists(ars_dir_path):
        shutil.rmtree(ars_dir_path)    
    os.mkdir(ars_dir_path)
    #adding a forward slash to the path
    ars_dir_path += "/"
    
    #loading the model and data split objects 
    data_splt_model_tuple = load_and_list_model_and_data_split_objects(folder_path_models, folder_path_data_splits)
    
    #calculating and writing the ARS and prediction results 
    detailed_perf_df = calculate_detailed_pred_ARS(data_splt_model_tuple, ars_dir_path)
    


# In[11]:


parent_folder = "/Users/Javad/Desktop/Karen/ARICleanRes/AllCalssifiersResults/ClassifiersPredDetails/"
folder_path_data_splits = "/Users/Javad/Desktop/Karen/ARICleanRes/DataSplits/"


results_folder_list = [x for x in os.walk(parent_folder)][0][1]
folder_path_models_list = [parent_folder+folder_name for folder_name in results_folder_list]

for folder_path_models in folder_path_models_list:
    print(folder_path_models)
    print("==================================================\n\n")

    try:
        driver_fun_extracting_ARS(folder_path_models + "/", folder_path_data_splits)
    except:
        print("Oops!", sys.exc_info()[0], "occurred.")
        print("Next entry.")
        print(folder_path_models)


# In[4]:


results_folder_list


# In[ ]:




