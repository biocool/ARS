# This script assumes that the ET data are in the same folder as this file and are in csv format.
# the two first lines will be used for running on a server (avoiding the no display name error )

# for making directory
import os
# working with data frames
import pandas as pd
# "train_test_split" function
from sklearn.model_selection import train_test_split
# for listing all files
import glob
# for pickling (saving) data split objects
import pickle


def read_et_dataset_and_train_test_split(full_file_path):
    """
    # Description:
    This function reads an ET dataset and does the train test split (test set is 20% of the whole dataset).

    # Args:
    "full_file_path": a string which is the full path for an eyetracking csv file (a feature matrix)

    # Returns:
    a tuple which is the output of the "train_test_split" (a built-in function)
    """

    # reading the csv file
    et_data = pd.read_csv(full_file_path, index_col=0)  # index_col=0 ==> has row name
    et_data = pd.DataFrame(et_data)

    # removing the non-Complete cases (those cases that includes NA)
    et_data.dropna(axis=0, how='any', inplace=True)

    # extracting the feature matrix and label vector
    X = et_data.iloc[:, :-1]
    y = et_data.iloc[:, -1]

    # return the data split tuple
    return train_test_split(X, y, test_size=0.2, random_state=6156)


def data_split_driver_function(full_file_path):
    """
    # Description:
    This function is the driver function for the whole process of data splitting.
    It reads all ET combination datasets which are csv files in the "full_file_path" folder.
    It uses "random_state=6156", which was used for the data spliting in the training process.

    # Args:
    "full_file_path": a string which is the full path for all eyetracking csv files.

    #NOTE: this folder must have a subfolder "/DataSplits"

    # Returns:
    Not explicit. It writes all data split objects as a dictionary (dict(zip(['X_train', 'X_validation', 'X_test', 'y_train', 'y_validation', 'y_test'], [X_train, X_validation, X_test, y_train, y_validation, y_test])))
    """

    # extracting the folder path
    path_to_folder = "/".join([item for item in full_file_path.split("/")[:-1]])

    # extracting the file name from the full path
    file_name = full_file_path.split("/")[-1][:-4]  # [:-4] all characters except the extension which is ".csv"
    print(file_name)

    # selecting discovery and test set:
    X_discovery, X_test, y_discovery, y_test = read_et_dataset_and_train_test_split(full_file_path)

    # selecting train and validation set:
    X_train, X_validation, y_train, y_validation = train_test_split(X_discovery, y_discovery, test_size=0.2,
                                                                    random_state=6156)

    # creating a dictionary from the data splits
    current_data_split_train_valid_test = dict(
        zip(['X_train', 'X_validation', 'X_test', 'y_train', 'y_validation', 'y_test'],
            [X_train, X_validation, X_test, y_train, y_validation, y_test]))

    # creating an appropriate file name for writing the data split objects
    full_file_path_for_writing = path_to_folder + "/DataSplits" + '/' + file_name
    full_file_name_for_data_split_train_valid_test_object = full_file_path_for_writing + "_Data_split_train_valid_test.dsplt "
    # print(full_file_name_for_data_split_train_valid_test_object)

    # saving the train/validation/test data split dict
    pickle_file = open(full_file_name_for_data_split_train_valid_test_object, 'wb')
    pickle.dump(current_data_split_train_valid_test, pickle_file)

    # writing all data as csv files (Karen prefers to have these files as well)
    X_train.to_csv(path_to_folder + "/DataSplits" + '/' + file_name + "_X_train.csv")
    y_train.to_csv(path_to_folder + "/DataSplits" + '/' + file_name + "_y_train.csv")

    X_validation.to_csv(path_to_folder + "/DataSplits" + '/' + file_name + "_X_validation.csv")
    y_validation.to_csv(path_to_folder + "/DataSplits" + '/' + file_name + "_y_validation.csv")

    X_test.to_csv(path_to_folder + "/DataSplits" + '/' + file_name + "_X_test.csv")
    y_test.to_csv(path_to_folder + "/DataSplits" + '/' + file_name + "_y_test.csv")



################################################################################################################################################################################################
#############################################################################################################################################################################################
# calling driver function

# folder containing all ET combination in csv format
folder_path = "/Users/Javad/Desktop/Karen/ARICleanRes/All_ET_DataCombination/*.csv"

# listing all csv files in the folder
all_csv_files = glob.glob(folder_path)

# extracting the folder path
path_to_folder = "/".join([item for item in all_csv_files[0].split("/")[:-1]])

# making a directory for writing the data split objects in a separate directory
os.mkdir(path_to_folder + "/DataSplits")

# applying driver function on all datasets
for file_full_path in all_csv_files:
    data_split_driver_function(file_full_path)
################################################################################################################################################################################################
