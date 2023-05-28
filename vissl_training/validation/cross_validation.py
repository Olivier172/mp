import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path
from termcolor import cprint
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from similarity_matrix import get_targets, compile_total_log
from embedding_gallery import read_embedding_gallery

def search_best_hyperparam(param_grid, classifier,train_data, train_labels, test_data, test_labels):
    """
    Searches the best hyperparemters for an svm classifier to be trained on top of
    train and test data.

    Args:
        param_grid (dict): dictionary or list of dictionaries that contain different hyperparams as keys and values to test in a list.
        classifier: the classifier to search the best hyperparams for.
        train_data (np.ndarray): training set.
        train_labels (np.ndarray): ground truth labels for training set.
        test_data (np.ndarray): testing set.
        test_labels (np.ndarray): ground truth labels for testing set.

    Returns:
        best_params: the best svm params from the gridsearch.
        test_score: the final testing score on the model with the best hyperparams on the test set.
    """
    #Create StratifiedKFold generator
    #5 folds with 1/5 test data and 4/5 training data to cross validate. 
    #The folds have equal class distribution in this case
    skf = StratifiedKFold(n_splits=5)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(classifier, param_grid, cv=skf)
    grid_search.fit(train_data, train_labels)

    # Get the best hyperparameters and corresponding score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Train a final model using the best hyperparameters
    final_model = type(classifier)(**best_params)
    final_model.fit(train_data, train_labels)

    # Evaluate the final model on the test set (optional)
    test_score = final_model.score(test_data, test_labels)
    
    return best_params, test_score

def search_best_hyperparam_svm(train_data, train_labels, test_data, test_labels):
    
    # Define the parameter grid
    param_grids = [
        #eval diff kernels and regularization params c
        {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 0.5, 1, 5, 10]
        },
        #eval different degrees of polynomials
        {
            "kernel": ["poly"],
            "degree" : [2, 3, 4, 5, 10],
            "C": [0.1, 0.5, 1, 5, 10]
        }
    ]
    
    # Create the SVM model
    svm = SVC()
    
    #search the best hyperparams
    best_params, test_score = search_best_hyperparam(param_grids, svm, train_data, train_labels, test_data, test_labels)
    
    return best_params, test_score

def search_best_hyperparam_mlp(train_data, train_labels, test_data, test_labels):
    
    # Define the parameter grid
    param_grid = {
        "hidden_layer_sizes": [(32), (64), (128), (128,64)],
        "solver": ["lbfgs"],
        "alpha": [0.01, 0.1],
        "max_iter" : [10_000]
    }

    # Create the MLP classifier
    mlp = MLPClassifier()
    
    #search the best hyperparams
    best_params, test_score = search_best_hyperparam(param_grid, mlp, train_data, train_labels, test_data, test_labels)
    
    return best_params, test_score
    
def get_train_test_sets(data, labels, strict=False, verbose=False):
    """
    Extracts train and test sets using sklearns train test split or strict split from json file if strict=True.

    Args:
        data (np.ndarray): data to be split up.
        labels (np.ndarray): ground truth labels for data.
        strict (bool, optional): Wether to use the strict split from the json file. Defaults to False.
        verbose (bool, optional): switch to allow prints. Defaults to False.

    Returns:
        train_set (np.ndarray): training data.
        train_labels (np.ndarray): labels for training data.
        test_set (np.ndarray): testing data.
        test_labels (np.ndarray): labels for testing data.
    """
    if(strict):
        json_file = Path("data/strict_train_test.json")
        if(verbose):
            cprint(f"Info: using the strict train test split from {json_file}", "yellow")
        # Read the JSON file
        with open(json_file, "r") as file:
            json_data = file.read()
        # Convert JSON to dictionary
        strict_train_test = json.loads(json_data)
        # Eligible classes: 
        if(verbose):
            print(f"Amount of classes that can be used for strict testing: {len(strict_train_test.keys())} / {len(np.unique(labels))}")
        #setup sets
        train_set = []
        train_labels = [] 
        test_set = []
        test_labels = []
        for cls in strict_train_test.keys():
            #Collect train queries
            train_queries = strict_train_test[cls]["train"]
            for query in train_queries:
                idx = query["gallery_idx"]
                train_query = data[idx]
                train_label = labels[idx]
                train_set.append(train_query)
                train_labels.append(train_label)
            
            #Collect test queries
            test_queries = strict_train_test[cls]["test"]
            for query in test_queries:
                idx = query["gallery_idx"]
                test_query = data[idx]
                test_label = labels[idx]
                test_set.append(test_query)
                test_labels.append(test_label)
        #convert to np arrays
        train_set = np.array(train_set)
        train_labels = np.array(train_labels)
        test_set = np.array(test_set)
        test_labels = np.array(test_labels)
        
        
    else:
        #the following code in comments unfortunately gave worse results, so we are not going to use it
        #FILTER: only use data that has at least 2 occurences. (so that at least one ends up in the training set and one in the test set)
        # unique_labels, counts = np.unique(labels, return_counts=True)# Find unique elements and their counts
        # stratify_class_labels = unique_labels[counts >= 2]# Filter elements that appear at least twice
        # data_valid = []
        # labels_data_valid = []
        # for idx, l in enumerate(labels):
        #     if(l in stratify_class_labels):
        #         data_valid.append(data[idx])
        #         labels_data_valid.append(l)
        # data_valid = np.array(data_valid)
        # labels_data_valid = np.array(labels_data_valid)
    
        if(verbose):
            cprint("Info: splitting data using standard sklearn train test split", "yellow")
            
        train_set, test_set, train_labels, test_labels= train_test_split(
            data,          #data
            labels,        #targets
            test_size=0.2, #20% test set, 80% train set
            random_state=0 #for reproducable results of the random shuffling
        )
        
    if(verbose):
        #Proportion of training data
        prop_train = (len(train_set)/ (len(train_set) + len(test_set)) ) * 100
        #Proportion of testing data
        prop_test = (len(test_set)/ (len(train_set) + len(test_set)) ) * 100
        print(f"Size of the training set {len(train_set)}\nThe training set contains {prop_train}% of the data\n")
        print(f"Size of the test set {len(test_set)}\nThe test set contains {prop_test}% of the data")
    
    return train_set, train_labels, test_set, test_labels


def log_results(output_file:Path, model_name:str, best_params:dict, test_score):
    """
    Logs cross validation results for this model with svm and mlp classifier

    Args:
        output_file (Path): path to output file
        model_name (str): name of the model used
        best_params (dict): dict with best params
        test_score: the test score for best params
    """
    #Creating message (lines)
    lines = [] 
    # Get the current date and time
    current_datetime = datetime.now()
    # Extract the date and time components
    current_date = current_datetime.date()
    current_time = current_datetime.time()
    # Convert to string format
    date_string = current_date.strftime('%Y-%m-%d')
    time_string = current_time.strftime('%H:%M:%S')
    lines.append("-"*90 + "\n")
    lines.append(f"cross validation logging results on {date_string} @ {time_string}.\n")
    lines.append(f"Embedding gallery of model {model_name} was used to calculated these scores.\n")
    #log results
    lines.append(f"best params {best_params}\n")
    lines.append(f"test score {test_score}\n")
    lines.append("-"*90 + "\n\n")
    
    #log results to file
    cprint(f"Logging cross validation results to a file on path : {output_file}", "green")
    with open(output_file, "w") as f:
        f.writelines(lines)
    #save best params in json format to use during testing phase
    json_file = output_file.parent / (output_file.stem  + "_best_params.json")  
    cprint(f"Logging best params to a file on path : {json_file}", "green")
    with open(json_file, "w") as f:
        json.dump(best_params, f)
        
    
def cross_validate(verbose=False, exist_ok=False):
    """
    Function to perform cross validation to tune hyperparameters of svm or mlp.

    Args:
        verbose (bool, optional): print control. Defaults to False.
        exist_ok (bool, optional): Wether to recalculate optimal hyperparams
                                   and overwrite previouse logfiles. Defaults to False.
    """
    gallery:torch.Tensor
    gallery_norm:torch.Tensor
    labels:list
    
    #get the models to calculate cross validation for
    targets = get_targets()
    
    #select which classifier to optimize hyperparams for
    classifier = input("please select a classifier to search optimal hyperparams for (SVM/mlp): ")
    if(classifier != "mlp"):
        classifier = "svm"
    else:
        classifier = "mlp"
        
    #determine output file names
    file_names = [f"cross_val_{classifier}.txt", f"cross_val_strict_{classifier}.txt"]    
         
    for target in targets:
        cprint(f"Calculating cross validation for target {target}", "red")
        p = Path(f"data/{target}")
        gallery, gallery_norm, labels = read_embedding_gallery(p)
        #Convert our data into numpy arrays
        data = gallery_norm.numpy()
        labels = np.array(labels)
        
        logfiles = [p.joinpath(fn) for fn in file_names]
        for logfile in logfiles:
            if(logfile.is_file() and not(exist_ok)):
                cprint(f"Info: logfile for this cross validation already exists at {logfile}, skip calculation", "yellow")
                continue
            #train test split
            if(logfile.name == f"cross_val_{classifier}.txt"):
                train_set, train_labels, test_set, test_labels = get_train_test_sets(data, labels, strict=False, verbose=verbose)
            elif(logfile.name == f"cross_val_strict_{classifier}.txt"):
                train_set, train_labels, test_set, test_labels = get_train_test_sets(data, labels, strict=True, verbose=verbose)
            
            if(classifier == "svm"):
                best_params, test_score= search_best_hyperparam_svm(train_set, train_labels, test_set, test_labels)
            elif(classifier == "mlp"):
                best_params, test_score = search_best_hyperparam_mlp(train_set, train_labels, test_set, test_labels)

            log_results(logfile, target, best_params, test_score)
            
    dir = Path("data")
    f_name = f"total_cross_val_log_{classifier}.txt"
    output_file = dir / f_name
    compile_total_log(
        output_file=output_file,
        dir=dir,
        targets=targets,
        file_names = file_names
    )
    
def main():
    cross_validate(verbose=True, exist_ok=False)
    

if __name__ == "__main__":
    main()