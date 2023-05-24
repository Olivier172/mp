import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path
from termcolor import cprint
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from similarity_matrix import get_targets

from embedding_gallery import read_embedding_gallery

def search_best_hyperparam(param_grid, classifier,train_data, train_labels, test_data, test_labels):
    """
    Searches the best hyperparemters for an svm classifier to be trained on top of
    train and test data.

    Args:
        param_grid (dict): 
        classifier:
        train_data (np.ndarray): _description_
        train_labels (np.ndarray): _description_
        test_data (np.ndarray): _description_
        test_labels (np.ndarray): _description_

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
    final_model = SVC(**best_params)
    final_model.fit(train_data, train_labels)

    # Evaluate the final model on the test set (optional)
    test_score = final_model.score(test_data, test_labels)
    
    return best_params, test_score

def search_best_hyperparam_svm(train_data, train_labels, test_data, test_labels):
    
    # Define the parameter grid
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': [0.1, 0.01]
    }
    
    # Create the SVM model
    svm = SVC()
    
    #search the best hyperparams
    best_params, test_score = search_best_hyperparam(param_grid, svm, train_data, train_labels, test_data, test_labels)
    
    return best_params, test_score

def search_best_hyperparam_mlp(train_data, train_labels, test_data, test_labels):
    
    # Define the parameter grid
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01]
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
        train_set: training data.
        train_labels: labels for training data.
        test_set: testing data.
        test_labels: labels for testing data.
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
        test_set = np.array(test_set)
        
    else:
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


def log_results(file_path:Path, best_params, test_score):
    lines = []
    lines.append("")  
    
def main():
    model_name:str
    gallery:torch.Tensor
    gallery_norm:torch.Tensor
    labels:list
    
    #get the models to calculate cross validation for
    targets = get_targets()
    
    for target in tqdm(targets):
        cprint(f"Calculating cross validation for target {target}", "red")
        p = Path("data/" + target)
        gallery, gallery_norm, labels = read_embedding_gallery(p)
        #Convert our data into numpy arrays
        data = gallery_norm.numpy()
        labels = np.array(labels)
        
        logfiles = [p.joinpath("cross_val.txt"), p.joinpath("cross_val_strict.txt")]
        for logfile in logfiles:
            #train test split
            if(logfile.name == "cross_val.txt"):
                train_set, train_labels, test_set, test_labels = get_train_test_sets(data, labels, strict=False, verbose=True)
            elif(logfile.name == "cross_val_strict.txt"):
                train_set, train_labels, test_set, test_labels = get_train_test_sets(data, labels, strict=True, verbose=True)
            
            best_params_svm, test_score_svm = search_best_hyperparam_svm(train_set, train_labels, test_set, test_labels)
            best_params_mlp, test_score_mlp = search_best_hyperparam_mlp(train_set, train_labels, test_set, test_labels)
    

if __name__ == "__main__":
    main()