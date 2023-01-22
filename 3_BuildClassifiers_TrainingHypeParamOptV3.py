# This script assumes that all the ET data are in the same folder as this file and are in csv format.
# the two first lines will be used for running on a server (avoiding the no display name error )

# for plotting ROC curve and confusion matrix
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
# for making directory
import os
# for preprocessing related
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.compose import make_column_transformer
# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import ExtraTreeClassifier
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# working with data frames
import pandas as pd
# for picking gridsearch objects
import joblib
# for listing all files
import glob

# pipeline related (Data split, grid search, and train test split)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
# Perf assessment
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import ConfusionMatrixDisplay


def read_et_dataset_and_train_test_split(full_file_path):
    """
    # Description:
    This function reads an ET dataset and does the train test split (test set is 20% of the whole dataset).

    #:argument:
    "full_file_path": a string which is the full path for an eyetracking csv file (a feature matrix)

    #:returns:
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


def build_preprocess_procedure(X_train_colnames_list):
    """
    # Description
    This function makes a pipeline using "make_pipeline" (built-in) and "build_preprocess_procedure" functions.
    # "X_train_colnames_list" a list of the col names for the training data
    # 'gender' ==> OneHotEncoder
    # Passthrough:
    # 'JA_04_BoxCycles',
    # 'JA_05_CombCycles',
    # 'JA_12_GreatScarfCycles',
    # 'JATotalNo_of_FullCycle',
    # + ages
    # rest : RobustScaler()


    #:argument:
    "X_train_colnames_list": a vector of the string which is the column name for the feature matrix.
    This vector will be passed to the "build_preprocess_procedure" for applying appropriate preprocessing on each feature.

    #:returns:
    "preprocessing_clf_pipeline" the pipeline object (contains the preprocessing and classification steps).
    Note: the default classifier here is random forest ("RandomForestClassifier")
    """

    whole_cols_4_robust_scaler = {'GeoPrefPcntFixGeo',
                                  'GeoPrefSacPerSecGeo',
                                  'GeoPrefSacPerSecSoc',

                                  'CmplxPcntFixGeo',
                                  'CmplxSacPerSecGeo',
                                  'CmplxSacPerSecSoc',

                                  'OutdoorPcntFixGeo',
                                  'OutdoorSacPerSecGeo',
                                  'OutdoorSacPerSecSoc',

                                  'TrafficPcntFixTraffic',
                                  'TrafficSacPerSecMthr',
                                  'TrafficSacPerSecTraffic',
                                  'TrafficPcntFixEyes',
                                  'TrafficPcntFixFace',

                                  'TechnoPcntFixTechno',
                                  'TechnoSacPerSecTechno',
                                  'TechnoSacPerSecMtr',
                                  'TechnoPcntFixEyes',
                                  'TechnoPcntFixFace',

                                  'JA_PcntFixFace',
                                  'JA_PcntFixTarget',
                                  'JA_PcntFixNonTarget'}

    # selecting the columns that should ondergo the scaler transformation

    cols_4_robust_scaler = set(X_train_colnames_list).intersection(whole_cols_4_robust_scaler)
    cols_4_robust_scaler_list = list(cols_4_robust_scaler)  # convert to list for using in the "make_column_transformer"

    one_hot_col_transformer_4_gender = (OneHotEncoder(), ['gender'])
    robust_scaler_transformer = (RobustScaler(), cols_4_robust_scaler_list)

    preprocessor = make_column_transformer(one_hot_col_transformer_4_gender, robust_scaler_transformer,
                                           remainder='passthrough')
    return preprocessor


def build_prep_clf_pipelines_list_param_grid_list(
        X_train_colnames_list):  # "X_train_colnames_list" a list of the col names for the training data
    """
    # Description
    This function makes a pipeline using "make_pipeline" (built-in) and "build_preprocess_procedure" functions.
    #Input:
    "X_train_colnames_list": a vector of the string which is the column name for the feature matrix.
    This vector will be passed to the "build_preprocess_procedure" for applying appropriate preprocessing on each feature.
    #Output:
    "(preprocessing_clf_pipeline_list, param_grid_list)" a tuple of two element:
    "preprocessing_clf_pipeline_list" a list of pipeline objects (each contains the preprocessing and classification steps).
    "param_grid_list" a list of dictionaries. Each dictionary is a parameters values to be used in grid search
    """

    preprocessor = build_preprocess_procedure(X_train_colnames_list)  # It is the same for all the classifiers.
    # a list of classifier with initialization (grid search will be done to tune hyperparameters)
    # setting "random_state" for reproducible results
    # n_jobs=-3 means all cores except 2
    my_random_state = 6156
    classifiers_list = [RandomForestClassifier(warm_start=True, random_state=6156, n_jobs=-3),
                        LogisticRegression(warm_start=True, random_state=6156, class_weight='balanced',
                                           solver='liblinear', n_jobs=-3),
                        KNeighborsClassifier(n_jobs=-3),
                        RadiusNeighborsClassifier(n_jobs=-3),
                        GaussianProcessClassifier(random_state=6156, max_iter_predict=500, n_restarts_optimizer=3,
                                                  n_jobs=-3),
                        QuadraticDiscriminantAnalysis(store_covariance=True),
                        LinearDiscriminantAnalysis(store_covariance=True),
                        NuSVC(probability=True, cache_size=500, random_state=6156),
                        DecisionTreeClassifier(random_state=6156),
                        # MLPClassifier(random_state=6156),
                        BayesianGaussianMixture(max_iter=1000, n_init=10, random_state=6156),
                        ComplementNB(),
                        ExtraTreeClassifier(random_state=6156),
                        GaussianMixture(n_init=10, max_iter=1000, random_state=6156),
                        LogisticRegressionCV(penalty='elasticnet', solver='saga', l1_ratios=0.5, n_jobs=-3,
                                             random_state=6156),
                        # Ensemble classifiers
                        AdaBoostClassifier(n_estimators=1000, random_state=6156),
                        GradientBoostingClassifier( random_state=6156)


                        ]

    preprocessing_clf_pipeline_list = [make_pipeline(preprocessor, clf) for clf in classifiers_list]

    # a list of parameters for the above classifiers to be used in grid search
    param_grid_list = [
        # RandomForestClassifier
        {
            'randomforestclassifier__n_estimators': [100, 500, 1000],
            'randomforestclassifier__max_depth': [2, 4, 8, 16],
            'randomforestclassifier__min_samples_split': [2, 5],
            'randomforestclassifier__min_samples_leaf': [1, 2],
            'randomforestclassifier__max_samples': [0.7, 0.8, 0.99]
        },
        # LogisticRegression
        {
            'logisticregression__penalty': ['l1', 'l2', 'elasticnet', 'none']
        },
        # KNeighborsClassifier
        {
            'kneighborsclassifier__n_neighbors': [3, 5, 10, 20],
            'kneighborsclassifier__weights': ['distance', 'uniform'],
            'kneighborsclassifier__p': [1, 2]
        },
        # RadiusNeighborsClassifier
        {
            'radiusneighborsclassifier__p': [1, 2],
            'radiusneighborsclassifier__outlier_label': ['nonASD']
        },
        # GaussianProcessClassifier
        {
            'gaussianprocessclassifier__warm_start': [True, False]
        },
        # QuadraticDiscriminantAnalysis
        {
            'quadraticdiscriminantanalysis__store_covariance': [True]
        },
        # LinearDiscriminantAnalysis
        {
            'lineardiscriminantanalysis__solver': ['svd', 'lsqr', 'eigen']

        },
        # NuSVC
        {
            'nusvc__nu': [0.5, 0.8, 0.1],
            'nusvc__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            'nusvc__degree': [3, 4, 5],
            'nusvc__gamma': ['scale', 'auto']
        },
        # DecisionTreeClassifier
        {
            'decisiontreeclassifier__criterion': ['gini', 'entropy', 'log_loss'],
            'decisiontreeclassifier__max_depth': [2, 4, 8, 16],
            'decisiontreeclassifier__min_samples_split': [2, 5],
            'decisiontreeclassifier__min_samples_leaf': [1, 2],

        },
        # BayesianGaussianMixture
        {
            'bayesiangaussianmixture__warm_start': [True, False],
            'bayesiangaussianmixture__covariance_type': ['full', 'tied', 'diag', 'spherical']
        },
        # ComplementNB
        {
            'complementnb__norm': [True, False]
        },
        # ExtraTreeClassifier
        {
            'decisiontreeclassifier__criterion': ['gini', 'entropy', 'log_loss'],
            'decisiontreeclassifier__max_depth': [2, 4, 8, 16],
            'decisiontreeclassifier__min_samples_split': [2, 5],
            'decisiontreeclassifier__min_samples_leaf': [1, 2],

        },
        # GaussianMixture
        {
            'gaussianmixture__n_components': [1, 2, 3],
            'gaussianmixture__covariance_type': ['full', 'tied', 'diag', 'spherical'],
            'gaussianmixture__warm_start': [True, False]

        },
        # LogisticRegressionCV
        {
            "logisticregressioncv__max_iter": [1000, 2000]

        },
        # Ensemble Classifiers

        # AdaBoostClassifier
        # for some example take a look at https://www.datacamp.com/tutorial/adaboost-classifier-python
        {
            # setting the base classifier type
            "adaboostclassifier__estimator": [
                DecisionTreeClassifier(random_state=6156, criterion='entropy', max_depth=2, min_samples_leaf=1,
                                       min_samples_split=2),
                DecisionTreeClassifier(random_state=6156, criterion='gini', max_depth=4, min_samples_leaf=1,
                                       min_samples_split=2),
                LinearDiscriminantAnalysis(store_covariance=True, solver='lsqr'),
                LinearDiscriminantAnalysis(store_covariance=True, solver='svd'),
                LogisticRegression(warm_start=True, random_state=6156, class_weight='balanced', solver='liblinear',
                                   n_jobs=-3, penalty='l1'),
                LogisticRegression(warm_start=True, random_state=6156, class_weight='balanced', solver='liblinear',
                                   n_jobs=-3, penalty='l1'),
                NuSVC(probability=True, cache_size=500, random_state=6156, degree=3, nu=0.8),
                NuSVC(probability=True, cache_size=500, random_state=6156, kernel='poly', nu=0.1),
                NuSVC(probability=True, cache_size=500, random_state=6156, kernel='rbf', nu=0.1),
                NuSVC(probability=True, cache_size=500, random_state=6156, kernel='sigmoid', nu=0.1),
                QuadraticDiscriminantAnalysis(store_covariance=True),
                ExtraTreeClassifier(random_state=6156, criterion='entropy', max_depth=2, min_samples_leaf=1,
                                    min_samples_split=2),
                ExtraTreeClassifier(random_state=6156, criterion='gini', max_depth=4, min_samples_leaf=1,
                                    min_samples_split=2)]

        },
        #GradientBoostingClassifier
        {
            'gradientboostingclassifier__n_estimators': [100, 500, 1000],
            'gradientboostingclassifier__max_depth': [3, 4, 5],
            'gradientboostingclassifier__min_samples_split': [2, 5],
            'gradientboostingclassifier__min_samples_leaf': [1, 2, 3],
            'gradientboostingclassifier__max_features': ['auto', 'sqrt', 'log2']
        },

    ]
    return (preprocessing_clf_pipeline_list, param_grid_list)


'''
# Description
This function makes a gridsearch object using "GridSearchCV" (built-in) and "build_pipeline" functions.
#Input: 
"X_train_colnames_list": a vector of the string which is the column name for the feature matrix.
This vector will be passed to the "build_pipeline" for making the pipline and mainly for applying appropriate preprocessing on each feature. 
#Output: 
"grid_search_list" a list of gridsearch objects. Each element has its own preprocessing+classifier pipeline and hyperparameter values "param_grid"
Note: the default classifier here is random forest
'''


def build_grid_search_4_all_clfs(
        X_train_colnames_list):  # "X_train_colnames_list" a list of the col names for the training data
    # building a tuple of lists using "build_prep_clf_pipelines_list_param_grid_list" function:
    # "preprocessing_clf_pipeline_list" a list of pipeline objects (each contains the preprocessing and classification steps).
    # "param_grid_list" a list of dictionaries. Each dictionary is a parameters values to be used in grid search

    preprocessing_clf_pipeline_list, param_grid_list = build_prep_clf_pipelines_list_param_grid_list(
        X_train_colnames_list)

    # comprehension
    grid_search_list = [GridSearchCV(pipeline,
                                     param_grid=param_grid,
                                     scoring=SCORING,  # it was  'balanced_accuracy' it also can be 'neg_log_loss'
                                     refit=True,
                                     verbose=2,
                                     cv=5,
                                     return_train_score=True
                                     ) for pipeline, param_grid in
                        zip(preprocessing_clf_pipeline_list, param_grid_list)]

    return grid_search_list


'''
# Description
This function writes the prediction performance including:
1- A csv file "classification_report" which shows:  precision, recall, f1-score for both ASD and nonASD classes as well as accuracy.
2- Visualized confusion matrix (pdf file)
3- ROC curve (pdf file)
#Input: 
"fitted_mode": trained model (fitted model) which we are going to assess it.
"X_validation" : test/validation feature matrix  
"y_validation": test/validation label vector 
"full_file_path": full path of the file which will be modified for writing the results (csv file, pdf files)
#Output: not explicit
'''


def write_prediction_results(fitted_mode, X_validation, y_validation, full_file_path_for_writing):
    ##############################################################################################################################
    # plotting ROC curve
    # preparing the file name
    full_file_path = full_file_path_for_writing + "_ROC.pdf"
    plt.figure()
    plot_roc_curve(fitted_mode, X_validation, y_validation, pos_label="ASD")
    plt.savefig(full_file_path, format="pdf", bbox_inches="tight")
    plt.close()
    # plt.show()
    ##############################################################################################################################
    # Plotting conf. Mat.
    # preparing the file name
    full_file_path = full_file_path_for_writing + "_ConfMatrix.pdf"
    # predicting the labels for the validatio set
    y_pred_validation = fitted_mode.predict(X_validation)
    # labels = ["nonASD","ASD"] --> it seems that the first class is the negative class, so the first output element is the TN
    cm = confusion_matrix(y_validation, y_pred_validation, labels=["nonASD", "ASD"])
    plt.figure()
    ConfusionMatrixDisplay(cm, display_labels=["nonASD", "ASD"]).plot()
    plt.savefig(full_file_path, format="pdf", bbox_inches="tight")
    plt.close('all')
    # plt.show()
    ##############################################################################################################################
    # writing classificatoin report to a file
    # calculating the classification performance on the validation set
    validatoin_classification_report = classification_report(y_validation, y_pred_validation, output_dict=True)
    # converting the classification performance to the apropriate dataframe
    validatoin_classification_report = pd.DataFrame(validatoin_classification_report).transpose()
    # writing the classification performance to a csv file
    full_file_path = full_file_path_for_writing + "classification_report.csv"
    validatoin_classification_report.to_csv(full_file_path)
    # closing all opened figs


'''
# Description:
This function is the driver  function for the whole process of doing grid search
and selecting the optimal classifier and writing the validation results for the given file. 
This file makes train/validation/test split, and uses the train for fitting the models and validation for the assessing the fitted model.
Finally saves the best model, according to the grid search result.   
# Input:
"full_file_path": a string which is the full path for an eyetracking csv file (a feature matrix)
# Output:
Not explicit 
'''


def driver_function(full_file_path):
    # printing a message
    path_to_folder = "/".join([item for item in full_file_path.split("/")[:-1]])
    file_name = full_file_path.split("/")[-1][:-4]  # [:-4] all characters except the extension which is ".csv"
    print(file_name)

    # selecting discovery and test set:
    X_discovery, X_test, y_discovery, y_test = read_et_dataset_and_train_test_split(full_file_path)

    # selecting train and validation set:
    X_train, X_validation, y_train, y_validation = train_test_split(X_discovery, y_discovery, test_size=0.2,
                                                                    random_state=6156)
    # extracting the column names of the training data
    X_train_colnames_list = list(X_train.columns)

    # building the gridsearch
    grid_search_list = build_grid_search_4_all_clfs(X_train_colnames_list)
    i = 0
    for grid_search in grid_search_list:
        print(grid_search)
        # fitting the grid_search
        grid_search.fit(X_train, y_train)
        full_file_path_for_writing = path_to_folder + "/Results" + classifier_name_list[i] + '/' + file_name
        full_file_path_for_writing = path_to_folder + "/Results" + SCORING + classifier_name_list[i] + '/' + file_name
        # saving the trained model
        joblib.dump(grid_search, full_file_name_for_grid_search_object)
        # applying the fitted model on the validation data and write the results
        write_prediction_results(grid_search, X_validation, y_validation, full_file_path_for_writing)
        i += 1


# In[ ]:


# # Reading all data set file names

folder_path = "./*.csv"
all_csv_files = glob.glob(folder_path)
path_to_folder = "/".join([item for item in all_csv_files[0].split("/")[:-1]])
# a global variable which is the list of the classifiers name
classifier_name_list = ["RandomForestClassifier",
                        "LogisticRegression",
                        "KNeighborsClassifier",
                        "RadiusNeighborsClassifier",
                        "GaussianProcessClassifier",
                        "QuadraticDiscriminantAnalysis",
                        "LinearDiscriminantAnalysis",
                        "NuSVC",
                        "DecisionTreeClassifier",  # "MLPClassifier",
                        "BayesianGaussianMixture",
                        "ComplementNB",
                        "ExtraTreeClassifier",
                        "GaussianMixture",
                        "LogisticRegressionCV",
                        "Adaboost",
                        "GradientBoostingClassifier"
                        ]

# making directory for writing the results of each classifier in a separate directory
SCORING = 'roc_auc'  # scoring function for selecting the best model in the gridsearch (objective function)
for i in range(len(classifier_name_list)):
    os.mkdir(path_to_folder + "/Results" + SCORING + classifier_name_list[i])

##applying driver function on all datasets##############################################################################################################################################################################################
for file_full_path in all_csv_files:
    driver_function(file_full_path)
################################################################################################################################################################################################
