import torch
import numpy as np
import json
from pathlib import Path
from termcolor import cprint
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from datetime import datetime

from similarity_matrix import get_targets #getting target models for performance calc
from cross_validation import get_train_test_sets #get train and test sets
from cross_validation import compile_total_log #the compile a complete log of all the logs per model
from embedding_gallery import read_embedding_gallery #reading in the embedding gallery for a certain model

def eval_performance(classifier, train_set, train_labels, test_set, test_labels):
    """
    Evaluate performance for the given estimator/classifier.
    Performance is measured by:
        -Training on the train_set and train_labels
        -Measuring accuracy for classification on the test_set with the test_labels
        -Calculate a confusion matrix
    
    Args:
        classifier : the classifier to evaluate.
        train_set (np.ndarray): the data to train on.
        train_labels (np.ndarray): ground truth for training data.
        test_set (np.ndarray): the data to test on.
        test_labels (np.ndarray): ground truth for testing data.
        
    Returns:
        accuracy : accuracy score from predictions on test set.
        cm: The confusion matrix.
    """
    #train on training set
    classifier.fit(train_set, train_labels) #retrain on training set
    
    #Calculate accuracy
    accuracy = classifier.score(test_set, test_labels) #calc acc score on testing set
    
    #Calculate confusion matrix
    y_true = test_labels.copy() #Ground truth
    y_pred = classifier.predict(test_set) #Predictions made by classifier
    #compute confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    #predicted class is the column nr, ground truth is the row nr
    #correct classifications can be found on the diagonal

    return accuracy, cm

def log_results(output_file:Path, model_name:str, accuracy, cm):
    """
    Logs cross validation results for this model with svm and mlp classifier

    Args:
        output_file (Path): path to output file.
        model_name (str): name of the model used.
        accuracy (dict): accuracy achieved on the testing set.
        cm: confusion matrix representing true and predicted labels.
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
    lines.append(f"performance assessment logging results on {date_string} @ {time_string}.\n")
    lines.append(f"Embedding gallery of model {model_name} was used to calculated these scores.\n")
    #log results
    lines.append(f"acc={accuracy*100}% (accuracy on testing set)\n")
    lines.append(f"cm= {cm}\n")
    lines.append("-"*90 + "\n\n")
    
    #Saving to file
    cprint(f"Logging cross validation results to a file on path : {output_file}", "green")
    with open(output_file, "a") as f:
        f.writelines(lines)

def performane_assesment(verbose:bool=False, exist_ok:bool=False):
    """
    Function to perform the performance assessment for an svm or mlp on a given embedding gallery.

    Args:
        verbose (bool, optional): print control. Defaults to False.
        exist_ok (bool, optional): wether to recalc the evaluation for models that already have a log file. Defaults to False.
    """
    gallery:torch.Tensor
    gallery_norm:torch.Tensor
    labels:list
    
    #get the models to calculate performance assessment for
    targets = get_targets("Choose a model to evaluate performace for")
    
    #select which classifier to optimize hyperparams for
    classifier = input("please select a classifier to search optimal hyperparams for: (SVM/mlp)")
    if(classifier != "mlp"):
        classifier = "svm"
    else:
        classifier = "mlp"
        
    #determine output file names
    file_names = [f"perf_eval_{classifier}.txt", f"perf_eval_strict_{classifier}.txt"]  
    
    for target in targets:
        cprint(f"Calculating performance assessment for target {target}", "red")
        p = Path(f"data/{target}")
        gallery, gallery_norm, labels = read_embedding_gallery(p)
        #Convert our data into numpy arrays
        data = gallery_norm.numpy()
        labels = np.array(labels)
        
        logfiles = [p.joinpath(fn) for fn in file_names]
        for logfile in logfiles:
            if(logfile.is_file() and not(exist_ok)):
                cprint(f"Info: logfile for this performance assessment already exists at {logfile}, skip calculation", "yellow")
                continue
            #train test split and reading in json_file path for file with best hyperparams
            if(logfile.name == f"perf_eval_{classifier}.txt"):
                train_set, train_labels, test_set, test_labels = get_train_test_sets(data, labels, strict=False, verbose=verbose)
                json_file = p / f"cross_val_{classifier}_best_params.json"
                
            elif(logfile.name == f"perf_eval_strict_{classifier}.txt"):
                train_set, train_labels, test_set, test_labels = get_train_test_sets(data, labels, strict=True, verbose=verbose)
                json_file = p / f"cross_val_strict_{classifier}_best_params.json"
            
            #read in best hyperparams from cross validation
            # Read the JSON file
            with open(json_file, "r") as file:
                json_data = file.read()
            # Convert JSON to dictionary
            best_params = json.loads(json_data)
            
            if(classifier == "svm"):
                #Create SVM estimator with the best hyperparams from cross validation
                if("degree" in best_params.keys()):
                    #degree is only in best_params if kernel was poly
                    svm = SVC(
                        C = best_params["C"],
                        kernel = best_params["kernel"],
                        degree = best_params["degree"] 
                    )
                else:
                    svm = SVC(
                        C = best_params["C"],
                        kernel = best_params["kernel"]
                    )
                acc, cm = eval_performance(svm, train_set, train_labels, test_set, test_labels)
            elif(classifier == "mlp"):
                #Create MLP estimator
                mlp = MLPClassifier(
                    hidden_layer_sizes = best_params["hidden_layer_sizes"],
                    solver= best_params["solver"],
                    max_iter=10_000
                )
                acc, cm = eval_performance(mlp, train_set, train_labels, test_set, test_labels)

            log_results(logfile, target, acc, cm)
            
    compile_total_log(
        output_file_name=f"total_perf_eval_log_{classifier}.txt", 
        dir=Path("data"),
        targets=targets,
        file_names = file_names
    )

def main():
    performane_assesment(verbose=True, exist_ok=False)
    
if __name__ == "__main__":
    main()