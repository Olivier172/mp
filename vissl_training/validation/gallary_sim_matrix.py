import torch 
import torchvision
import torchvision.transforms as transforms
import numpy as np
from datetime import datetime
from pathlib import Path
from termcolor import cprint
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve 

def read_embedding_gallary(dir:Path):
    cprint("In function read_embedding_gallary()", "green")
    try:
        fts_stack = torch.load(dir / "embedding_gallary.torch")
        print(f"fts_stack has shape {fts_stack.shape}")
        fts_stack_norm = torch.load(dir / "embedding_gallary_norm.torch")
        print(f"fts_stack_norm has shape {fts_stack_norm.shape}")
        labels = list()
        with open(dir / "embedding_gallary_labels.txt", "r") as f:
            labels = f.read().splitlines()
            print(f"labels list has length "+ str(len(labels)))
    except Exception:
        cprint(f"Unable to read embedding gallary, check paths (dir={dir})", "red")
        
    return fts_stack, fts_stack_norm, labels

def calc_ip_cosine_sim(fts_stack:torch.Tensor, dir:Path=None):
    """
    Calulates a simularity matrix based on the inner product / cosine simularity a metric.
    Args:
        fts_stack (torch.Tensor): fts_stack: stack of embeddings -> inner product
        fts_stack_norm (torch.Tensor): stack of normalized embeddings -> cosim

    Returns:
        torch.Tensor: The simularity matrix
    """
    #Calculate simularity from each embedding with every other embedding by doing a matrix product:
    sim_matrix = fts_stack.matmul(fts_stack.T)
    #statistics
    #print statistics data
    print(f"Max: {sim_matrix.max()}")
    print(f"Min: {sim_matrix.min()}")
    print(f"mean: {torch.mean(sim_matrix)}")
    print(f"std: {torch.std(sim_matrix)}")
    return sim_matrix

def calc_eucl_dist_sim(fts_stack:torch.Tensor, dir:Path=None):
    """
    Calulates a simularity matrix based on the euclidian distance as a metric.

    Args:
        fts_stack (torch.Tensor): A torch stack with all the embeddings (normalized or not)

    Returns:
        torch.Tensor: The simularity matrix
    """
    eucl_dists = [] 
    for tensor in fts_stack:
        d = [] #store all distances from this tensor to all the other tensors
        for other_tensor in fts_stack:
            d_to = (tensor - other_tensor).pow(2).sum().sqrt() #d(tensor, other_tensor)=euclid distance
            d.append(d_to)
        d = torch.tensor(d)
        #print("distance tensor has shape {}".format(d.shape))
        #add tensor to euclidian distances 
        eucl_dists.append(d)
    sim_matrix = torch.stack(eucl_dists)
    #statistics
    print(f"Max: {sim_matrix.max()}")
    print(f"Min: {sim_matrix.min()}")
    print(f"mean: {torch.mean(sim_matrix)}")
    print(f"std: {torch.std(sim_matrix)}")
    return sim_matrix
    
def calc_mAP(sim_matrix:torch.Tensor, labels, verbose=False):
    """
    Function to calculate the mean Average Precision of a simularity matrix

    Args:
        sim_matrix (torch.Tensor): The simularity matrix to calculate the mAP of.
        labels (list): list of labels containing the GT for every row in the simularity matrix
        verbose (bool, optional): Boolean switch to enable prints. Defaults to False.

    Returns:
        float: The mAP score.
    """
    cprint("In function calc_mAP()", "green")
    sim_matrix_np = sim_matrix.numpy()
    labels_np = np.array(labels)
    if(verbose):
        print(f"sim_matrix_np shape {sim_matrix_np.shape}")
        print(f"labels shape {labels_np.shape}")
        print(f"len sim_matrix_np {len(sim_matrix_np)}")
    
    
    #return indicis of all unique class labels
    classes = list(np.unique(labels_np))
    if(verbose):
        print(f"amount of classes {len(classes)}")
    #dictionary to store average precisions for every query
    AP_queries = {}
    for cls in classes:
        #init every class key with an empty list
        AP_queries[cls]=[]

    #calculate AP for every query in the sim_matrix
    for i in range(len(sim_matrix_np)):
        #y_true contains the ground truth for classification (True if label of query is the same)
        #gallary labels == query label (query is on the diagonal so the i th element of the i th row)
        query_label = labels_np[i]
        y_true = labels_np == query_label #find all matches for the label of the current query (boolean np array)
        y_score = sim_matrix_np[i] 
        #compute AP
        AP_query = average_precision_score(y_true=y_true, y_score=y_score)
        AP_queries[query_label].append(AP_query) #save average precision for this label with the results of it's class
    
    #AP is average precision of a class with different threshods (poisitions in the PR curve)    
    #calculate the AP for every class:
    AP = {}
    for k in AP_queries.keys():
        #compute average for this class and store in AP dict:
        AP[k]= sum(AP_queries[k]) / len(AP_queries[k])
        
    #calculate mAP, mean over all AP for each class    
    mAP = sum(AP.values()) / len(AP.values())
    
    if(verbose):
        cprint(f"mAP of this model is: {mAP}","magenta")
        
    return mAP
    
def log_mAP_scores(dir: Path, model_name:str, mAPs:dict):
    """
    Function to log mAP scores to a file. Results are appended to dir/mAP_scores.txt
    
    Args:
        dir (Path): directory to place logfile.
        model_name (str): name of the model used.
        mAPs: dictionary with mAP scores.
    """  
    p = dir / "mAP_scores.txt"
    cprint(f"Logging mAP scores to a file on path : {p}", "green")
    file = open(p, "a")
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
    lines.append(f"mAP Logging results on {date_string} @ {time_string}.\n")
    lines.append(f"Embedding library of model {model_name} was used to calculated these scores.\n")
    lines.append(f"inner product (ip): mAP={mAPs['ip']} \n")
    lines.append(f"cosine simularity (cosim): mAP={mAPs['cosim']} \n")
    lines.append(f"euclidian distance (eucl_dist): mAP={ mAPs['eucl_dist'] } \n")
    lines.append(f"euclidian distance with normalized features (eucl_dist_norm): mAP={mAPs['eucl_dist_norm']}\n")
    lines.append("-"*90 + "\n\n")
    file.writelines(lines)
    file.close()
        
def calc_sim_matrix(model_name:str, verbose=False, exist_ok=False, log_results=False):
    """
    Calulates the simularity matrix of the entire embedding gallary.
    This is done using 4 different metrics of simularity. 
    A mean Average Precision score is also calculated using the simularity matrix.
    
    Args:
        model_name (str): The name of the model to calculate the simularity matrix for based on the 
                          embedding gallary for this model.
        exist_ok (Bool): Boolean switch to recalculate and overwrite sim_matrix if true.
        verbose (Bool): Boolean switch to allow prints of this function.
        log_results (Bool): Boolean switch to log mAP scores to a logfile.
    """
    cprint("In function calc_sim_matrix()", "green")
    fts_stack:torch.Tensor
    fts_stack_norm:torch.Tensor
    #read input gallary for this model
    dir = Path("data/" + model_name)
    fts_stack, fts_stack_norm, labels = read_embedding_gallary(dir)
    #dictionary to save mAP scores for each metric
    mAPs = {
        "ip": 0.0, "cosim": 0.0, "eucl_dist": 0.0, "eucl_dist_norm": 0.0 
    }
    
    ### INNER PRODUCT AS A SIMULARITY METRIC ###
    cprint("\ninner product", "cyan")
    p = dir / "sim_mat_ip.torch"
    if( p.exists() and not(exist_ok)):
        cprint("sim_mat already exists","green")
        sim_mat = torch.load(p)
    else:
        cprint("sim_mat doesn't exist yet or exist_ok=true, calulating...", "yellow")
        sim_mat = calc_ip_cosine_sim(fts_stack=fts_stack)
        torch.save(sim_mat, p)
    mAPs["ip"] = calc_mAP(sim_matrix=sim_mat, labels=labels, verbose=verbose) 
    
    ### COSINE SIMULARITY AS A SIMULARITY METRIC ###
    cprint("\ncosine simularity", "cyan")
    p = dir / "sim_mat_cosim.torch"
    if( p.exists() and not(exist_ok)):
        cprint("sim_mat already exists","green")
        sim_mat = torch.load(p)
    else:
        cprint("sim_mat doesn't exist yet or exist_ok=true, calulating...", "yellow")
        sim_mat = calc_ip_cosine_sim(fts_stack=fts_stack_norm)
        torch.save(sim_mat, p)
    mAPs["cosim"] = calc_mAP(sim_matrix=sim_mat, labels=labels, verbose=verbose) 
       
    ### EUCIDIAN DISTANCE AS A SIMULARITY METRIC ###
    cprint("\neuclidian distance", "cyan")
    p = dir / "sim_mat_eucl_dist.torch"
    if( p.exists() and not(exist_ok)):
        cprint("sim_mat already exists","green")
        sim_mat = torch.load(p)
    else:
        cprint("sim_mat doesn't exist yet or exist_ok=true, calulating...", "yellow")
        sim_mat = calc_eucl_dist_sim(fts_stack=fts_stack)
        torch.save(sim_mat, p)
    #reverse scores in simularity matrix for mAP calculation (low euclidian distance = high score and vice versa)
    sim_mat = sim_mat*-1
    mAPs["eucl_dist"] = calc_mAP(sim_matrix=sim_mat, labels=labels, verbose=verbose) 
    
    ### EUCLIDIAN DISTANCE (NORMALIZED FEATURES) AS A SIMULARITY METRIC ###
    cprint("\neuclidian distance with normalized features", "cyan")
    p = dir / "sim_mat_eucl_dist_norm.torch"
    if( p.exists() and not(exist_ok)):
        cprint("sim_mat already exists","green")
        sim_mat = torch.load(p)
    else:
        cprint("sim_mat doesn't exist yet or exist_ok=true, calulating...", "yellow")
        sim_mat = calc_eucl_dist_sim(fts_stack=fts_stack_norm)
        torch.save(sim_mat, p)
    #reverse scores in simularity matrix for mAP calculation (low euclidian distance = high score and vice versa)
    sim_mat = sim_mat*-1
    mAPs["eucl_dist_norm"] = calc_mAP(sim_matrix=sim_mat, labels=labels, verbose=verbose) 
    
    if(verbose):
        print(mAPs)
        
    if(log_results):
        log_mAP_scores(dir, model_name, mAPs)
        
def calc_sim_matrices(model_names:list, verbose=False, exist_ok=False, log_results=False):
    """
    Calulates the simularity matrices of the entire embedding gallary of multiple models (listed in model_names).

    Args:
        model_names (list): The names of the models to calculate the simularity matrices for.
        exist_ok (Bool): Boolean switch to recalculate and overwrite sim_matrix if true.
        verbose (Bool): Boolean switch to allow prints of this function.
        log_results (Bool): Boolean switch to log mAP scores to a logfile.
    """ 
    for model_name in model_names:
        if(verbose):
            cprint(f" \nCalculating simularity matrix and mAP scores for model :{model_name}", "green")
        calc_sim_matrix(model_name, verbose=True, exist_ok=False, log_results=True)
    
def main():
    options = ["rotnet", "jigsaw", "moco32", "moco64", "simclr", "swav", "imgnet_pretrained", "all"]
    print(f"Choose a model to calculate simularity with the embeddinig gallary. Your options are: {options}")
    model_name = input("Your Choice:")
    while model_name not in options:
        print(f"Invalid option. Your options are: {options}")
        model_name = input("Your Choice:") 
    
    if(model_name == "all"):
        #all available models will be used:
        model_names = ["rotnet", "jigsaw", "moco32", "simclr", "imgnet_pretrained"]
        calc_sim_matrices(model_names, verbose=True, exist_ok=True, log_results=True)
    else:
        calc_sim_matrix(model_name, verbose=True, exist_ok=False, log_results=True)
        
if __name__ == "__main__":
    main()