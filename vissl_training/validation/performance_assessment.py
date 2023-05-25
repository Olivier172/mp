import torch
import numpy as np
import json
from pathlib import Path
from termcolor import cprint
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, average_precision_score
from datetime import datetime

from similarity_matrix import get_targets #getting target models for performance calc
from similarity_matrix import compile_total_log #the compile a complete log of all the logs per model
from cross_validation import get_train_test_sets #get train and test sets
from embedding_gallery import read_embedding_gallery #reading in the embedding gallery for a certain model


def calc_mAP(classifier, test_set, test_labels):
    """
    Calculates the mean average precision (mAP) for the classifier given.
    Probability scores are calculated on the test_set with the test_labels as ground truth
    classification labels.

    Args:
        classifier (sklearn clf): the classifier to use (svm or mlp)
        test_set (np.ndarray): test set to use for calculating probabilities.
        test_labels (np.ndarray): ground truth classifcation labels for test set.

    Returns:
        mAP (float): the mAP score.
    """
    y_probas = classifier.predict_proba(test_set) #get confidence scores for each class
    y_proba_labels = classifier.classes_ #get corresponding classes 
    
    #dictionary to store average precisions for every query
    AP_queries = {}
    #calculate AP for every query in the sim_matrix
    for idx, y_proba in enumerate(y_probas):
        #y_true contains the ground truth for classification (True if label of gt_label is the same)
        gt_label = test_labels[idx] #ground truth label / class name
        y_true = y_proba_labels == gt_label #true at the position of y_proba should be the highest to classify the same label as gt_label
        
        #compute AP for this query (with prediction score in y_proba and gt y_true)
        AP_query = average_precision_score(y_true=y_true, y_score=y_proba)
        
        #if the class of this query is not registered in AP_queries, register the key and init with empty list
        if(gt_label not in AP_queries.keys()):
            #init every class key with the first AP element calculated for that class
            AP_queries[gt_label]=[AP_query]
        else:
            AP_queries[gt_label].append(AP_query) #save average precision for this label with the results of it's class
    
    #AP is average precision of a class with different threshods (positions in the PR curve)    
    #calculate the AP for every class:
    AP = {}
    for k in AP_queries.keys():
        #compute average for this class and store in AP dict:
        AP[k]= sum(AP_queries[k]) / len(AP_queries[k])
        
    #calculate mAP, mean over all AP for each class    
    mAP = sum(AP.values()) / len(AP.values())
    
    return mAP

def eval_performance(classifier, train_set, train_labels, test_set, test_labels, verbose=False):
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
        verbose (bool, optional): print switch.
        
    Returns:
        accuracy (float): accuracy score from predictions on test set.
        mAP (float): the mean average precision score.
        cm: The confusion matrix.
    """
    #train on training set
    classifier.fit(train_set, train_labels) #retrain on training set
    
    #Calculate accuracy
    accuracy = classifier.score(test_set, test_labels) #calc acc score on testing set
    
    if(verbose):
        cprint(f"In function eval_performance", "yellow")
        print(f"Accuracy calculated for this classifier is {accuracy}")
        
    #Calculate confusion matrix
    y_true = test_labels.copy() #Ground truth
    y_pred = classifier.predict(test_set) #Predictions made by classifier
    #compute confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    #predicted class is the column nr, ground truth is the row nr
    #correct classifications can be found on the diagonal
    
    #calc mean average precision score
    mAP = calc_mAP(classifier=classifier, test_set=test_set, test_labels=test_labels)
    
    if(verbose):
        print(f"mAP calculated for this classifier is {mAP}")
        
    return accuracy, mAP, cm

def log_results(output_file:Path, model_name:str, accuracy, mAP, cm):
    """
    Logs cross validation results for this model with svm and mlp classifier

    Args:
        output_file (Path): path to output file.
        model_name (str): name of the model used.
        accuracy (float): accuracy achieved on the testing set.
        mAP (float): mAP score achieved on the testing set.
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
    lines.append(f"mAP={mAP*100}% (accuracy on testing set)\n")
    lines.append(f"cm= {cm}\n")
    lines.append("-"*90 + "\n\n")
    
    #Saving to file
    cprint(f"Logging performance results to a file on path : {output_file}", "green")
    with open(output_file, "a") as f:
        f.writelines(lines)

def performance_assesment(verbose:bool=False, exist_ok:bool=False):
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
            
            if(verbose):
                cprint(f"Info: creating {classifier} with best_params={best_params}", "yellow")
            if(classifier == "svm"):
                #Create SVM estimator with the best hyperparams from cross validation
                if("degree" in best_params.keys()):
                    #degree is only in best_params if kernel was poly
                    svm = SVC(
                        C = best_params["C"],
                        kernel = best_params["kernel"],
                        degree = best_params["degree"],
                        probability=True #calc class probabilities 
                    )
                else:
                    svm = SVC(
                        C = best_params["C"],
                        kernel = best_params["kernel"],
                        probability=True #calc class probabilities 
                    )
                acc, mAP, cm = eval_performance(svm, train_set, train_labels, test_set, test_labels, verbose=verbose)
            elif(classifier == "mlp"):
                #Create MLP estimator
                mlp = MLPClassifier(
                    hidden_layer_sizes = best_params["hidden_layer_sizes"],
                    solver= best_params["solver"],
                    max_iter=10_000
                )
                acc, mAP, cm = eval_performance(mlp, train_set, train_labels, test_set, test_labels, verbose=verbose)

            log_results(
                output_file=logfile, 
                model_name=target, 
                accuracy=acc, 
                mAP=mAP, 
                cm=cm
            )
            
    dir = Path("data")
    f_name = f"total_perf_eval_log_{classifier}.txt"
    output_file = dir / f_name
    compile_total_log(
        output_file=output_file, 
        dir=dir,
        targets=targets,
        file_names = file_names
    )

def main():
    performance_assesment(verbose=True, exist_ok=False)
    
if __name__ == "__main__":
    main()