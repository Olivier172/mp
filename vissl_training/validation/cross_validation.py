import numpy as np
import torch
from pathlib import Path
from termcolor import cprint
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

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
    
def main():
    model_name:str
    gallery:torch.Tensor
    gallery_norm:torch.Tensor
    labels:list
    
    model_name = "moco64"
    p = Path("data/" + model_name)
    gallery, gallery_norm, labels = read_embedding_gallery(p)
    #Convert our data into numpy arrays
    data = gallery_norm.numpy()
    labels = np.aray(labels)

if __name__ == "__main__":
    main()