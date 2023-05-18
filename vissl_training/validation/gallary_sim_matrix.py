import torch 
import numpy as np
from datetime import datetime
from pathlib import Path
from termcolor import cprint
from sklearn.metrics import average_precision_score

def read_embedding_gallary(dir:Path, embedding_gallary_name:str):
    """
    Reads in the embedding gallary from disk.

    Args:
        dir (Path): Path to the directory where the gallary is saved.
        embedding_gallary_name (str): The name of the embedding gallary used. 
                                      This can be "embedding_gallary" or "embedding_gallary_avg" 

    Returns:
        embedding_gallary: The embedding gallary contains a stack of embeddings for which the label is known.
        embedding_gallary_norm: Gallary with normalized embeddings.
        labels: Ground turth labels for every row (embedding) in the embedding gallary.
    """
    cprint("In function read_embedding_gallary()", "green")
    
    file_name = embedding_gallary_name + ".torch" 
    embedding_gallary = torch.load(dir / file_name)
    print(f"fts_stack has shape {embedding_gallary.shape}")
    
    file_name = embedding_gallary_name + "_norm.torch"
    embedding_gallary_norm = torch.load(dir / file_name)
    print(f"fts_stack_norm has shape {embedding_gallary_norm.shape}")
    labels = list()
    
    file_name = embedding_gallary_name + "_labels.txt"
    with open(dir / file_name, "r") as f:
        labels = f.read().splitlines()
        print(f"labels list has length "+ str(len(labels)))
     
    return embedding_gallary, embedding_gallary_norm, labels

def calc_ip_cosim(query_stack:torch.Tensor, embedding_gallary:torch.Tensor, verbose=False):
    """
    Calulates a simularity matrix based on the inner product / cosine simularity as a metric.
        -If you use normalized features/embeddings for the query_stack and embedding gallary, the cosine simularity (cosim) matrix 
        is calculated.
        -If you use unnormalized features/embeddings for the query_stack and embedding gallary, the inner product (ip) matrix 
        is calculated.
        
    Args:
        query_stack (torch.Tensor): This is a stack of all query embeddings to match to the embedding gallary.
        embedding_gallary (torch.Tensor): This is the embedding gallary itself. Each row consist of one embedding
        for which the label (ground truth) is known. The query embeddings are matched to these gallary embeddings.

    Returns:
        sim_matrix (torch.Tensor): The simularity matrix. This matrix has in each row simularity scores for one query
        from the query stack to all embeddings in the embedding gallary.
    """
    #Calculate simularity from each embedding of the query_stack with every embedding from the embedding gallary by doing a matrix product:
    sim_matrix = query_stack.matmul(embedding_gallary.T)
    if(verbose):
        #statistics
        print(f"Max: {sim_matrix.max()}")
        print(f"Min: {sim_matrix.min()}")
        print(f"mean: {torch.mean(sim_matrix)}")
        print(f"std: {torch.std(sim_matrix)}")
    return sim_matrix

def calc_eucl_dist_sim(query_stack:torch.Tensor, embedding_gallary:torch.Tensor, verbose=False) -> torch.Tensor:
    """
    Calulates a simularity matrix based on the euclidian distance as a metric.

    Args:
        query_stack (torch.Tensor): This is a stack of all query embeddings to match to the embedding gallary.
        embedding_gallary (torch.Tensor): This is the embedding gallary itself. Each row consist of one embedding
        for which the label (ground truth) is known. The query embeddings are matched to these gallary embeddings.

    Returns:
        sim_matrix (torch.Tensor): The simularity matrix. This matrix has in each row simularity scores for one query
        from the query stack to all embeddings in the embedding gallary.
    """
    eucl_dists = [] 
    for tensor in query_stack:
        d = [] #store all distances from this tensor to all the other tensors
        for other_tensor in embedding_gallary:
            d_to = (tensor - other_tensor).pow(2).sum().sqrt() #d(tensor, other_tensor)=euclid distance
            d.append(d_to)
        d = torch.tensor(d)
        #print("distance tensor has shape {}".format(d.shape))
        #add tensor to euclidian distances 
        eucl_dists.append(d)
    sim_matrix = torch.stack(eucl_dists)
    if(verbose):
        #statistics
        print(f"Max: {sim_matrix.max()}")
        print(f"Min: {sim_matrix.min()}")
        print(f"mean: {torch.mean(sim_matrix)}")
        print(f"std: {torch.std(sim_matrix)}")
    return sim_matrix
    
def calc_mAP(sim_matrix:torch.Tensor, gallary_labels:list , query_labels:list, embedding_gallary_name:str, verbose=False):
    """
    Function to calculate the mean Average Precision of a simularity matrix

    Args:
        sim_matrix (torch.Tensor): The simularity matrix to calculate the mAP of.
        gallary_labels (list): list of labels containing the class of every column in the simularity matrix 
                               (every class of the embedding gallary)
        query_labels (list): list of labels containing the ground truth classification label for every row in 
                             the simularity matrix (and each row contains the simularity scores for a query). 
        verbose (bool, optional): Boolean switch to enable prints. Defaults to False.

    Returns:
        mAP (float): The mAP score.
    """
    cprint("In function calc_mAP()", "green")
    #Convert everything to numpy
    sim_matrix_np = sim_matrix.numpy()
    gallary_labels_np = np.array(gallary_labels)
    query_labels_np = np.array(query_labels)
    if(verbose):
        print(f"sim_matrix_np shape {sim_matrix_np.shape}")
        print(f"gallary_labels shape {gallary_labels_np.shape}")
        print(f"query_labels shape {query_labels_np.shape}")
        print(f"len sim_matrix_np {len(sim_matrix_np)}")
    
    
    #return indicis of all unique class labels
    classes = list(np.unique(gallary_labels_np))
    if(verbose):
        print(f"amount of classes {len(classes)}")
        
    #dictionary to store average precisions for every query
    AP_queries = {}
    #calculate AP for every query in the sim_matrix
    for i in range(len(sim_matrix_np)):
        #y_true contains the ground truth for classification (True if label of query is the same)
        #gallary labels == query label 
        query_label = query_labels_np[i]
        y_true = gallary_labels_np == query_label #find all matches for the label of the current query (boolean np array)
        y_score = sim_matrix_np[i] 
        if(embedding_gallary_name == "embedding_gallary"):
            #in this case, the query embedding has a perfect match in the gallary
            #we need to remove this one before evaluating AP to prevent "easy match"
            y_true[i] = False #Remove perfect match
            y_score[i]= -99999.0 #large negative number to indicate no match
        #skip queries who have no positives:
        if(y_true.sum() == 0): #y_true.sum() counts the amount of True's
            continue
        #compute AP
        AP_query = average_precision_score(y_true=y_true, y_score=y_score)
        #if the class of this query is not registered in AP_queries, register the key and init with empty list
        if(query_label not in AP_queries.keys()):
            #init every class key with an empty list
            AP_queries[query_label]=[]
        AP_queries[query_label].append(AP_query) #save average precision for this label with the results of it's class
    
    if(verbose):
        print(f"amount of classes where an AP could be calculated {len(AP_queries)}")
        
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
    
def log_mAP_scores(dir: Path, model_name:str, embedding_gallary_name:str, mAPs:dict):
    """
    Function to log mAP scores to a file. Results are appended to dir/mAP_scores.txt
    
    Args:
        dir (Path): directory to place logfile.
        model_name (str): name of the model used.
        embedding_gallary_name (str): The name of the embedding gallary used. 
                                      This can be "embedding_gallary" or "embedding_gallary_avg" 
        mAPs: dictionary with mAP scores.
    """  
    file_name = embedding_gallary_name + "_mAP_scores.txt"
    p = dir / file_name
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
        
def calc_sim_matrices(model_name:str, embedding_gallary_name:str, verbose=False, exist_ok=False, log_results=False):
    """
    Calulates the simularity matrices of the entire embedding gallary.
    This is done using 4 different metrics of simularity. (Generating 4 simularity matrices) 
    A mean Average Precision score is also calculated using the simularity matrix.
    
    Args:
        model_name (str): The name of the model to calculate the simularity matrix for based on the 
                          embedding gallary for this model.
        embedding_gallary_name (str): The name of the embedding gallary used. 
                                      This can be "embedding_gallary" or "embedding_gallary_avg" 
        exist_ok (Bool): Boolean switch to recalculate and overwrite sim_matrix if true.
        verbose (Bool): Boolean switch to allow prints of this function.
        log_results (Bool): Boolean switch to log mAP scores to a logfile.
    """
    if(verbose):
        cprint("In function calc_sim_matrix()", "green")
    embedding_gallary:torch.Tensor
    embedding_gallary_norm:torch.Tensor
    gallary_labels:list
    query_labels:list
    
    #Reading in the embedding gallary
    dir = Path("data/" + model_name) #Directory to find gallaries for this model
    embedding_gallary, embedding_gallary_norm, gallary_labels = read_embedding_gallary(dir, embedding_gallary_name)
    
    #Reading in the query stack
    if(embedding_gallary_name == "embedding_gallary"):
        #in this case we query all the embeddings from the embedding gallary itself as a test set (the entire cornershop dataset embeddings)
        query_stack = embedding_gallary.clone().detach()
        query_stack_norm = embedding_gallary_norm.clone().detach()
        query_labels = gallary_labels.copy()
    else: #embedding_gallary_name == embedding_gallary_avg
        #in this case we query all the embeddings from the cornershop dataset (which are stored in the standard embedding gallary)
        #but the embedding gallary itself contains different embeddings (averages per class)
        query_stack, query_stack_norm, query_labels = read_embedding_gallary(dir, "embedding_gallary")
        
    #Dictionary to save mAP scores for each metric
    mAPs = {
        "ip": 0.0, "cosim": 0.0, "eucl_dist": 0.0, "eucl_dist_norm": 0.0 
    }
    
    ### INNER PRODUCT AS A SIMULARITY METRIC ###
    if(verbose):
        cprint("\ninner product", "cyan")
    file_name = embedding_gallary_name + "_sim_mat_ip.torch"
    p = dir / file_name
    if( p.exists() and not(exist_ok)):
        if(verbose):
            cprint("sim_mat already exists","green")
        sim_mat = torch.load(p)
    else:
        if(verbose):
            cprint("sim_mat doesn't exist yet or exist_ok=true, calulating...", "yellow")
        sim_mat = calc_ip_cosim(query_stack=query_stack, embedding_gallary=embedding_gallary, verbose=verbose)
        torch.save(sim_mat, p)
    mAPs["ip"] = calc_mAP(
        sim_matrix=sim_mat, 
        gallary_labels=gallary_labels, 
        query_labels=query_labels, 
        embedding_gallary_name=embedding_gallary_name,
        verbose=verbose
    ) 
    
    ### COSINE SIMULARITY AS A SIMULARITY METRIC ###
    if(verbose):
        cprint("\ncosine simularity", "cyan")
    file_name = embedding_gallary_name + "_sim_mat_cosim.torch"
    p = dir / file_name
    if( p.exists() and not(exist_ok)):
        if(verbose):
            cprint("sim_mat already exists","green")
        sim_mat = torch.load(p)
    else:
        if(verbose):
            cprint("sim_mat doesn't exist yet or exist_ok=true, calulating...", "yellow")
        sim_mat = calc_ip_cosim(query_stack=query_stack_norm, embedding_gallary=embedding_gallary_norm)
        torch.save(sim_mat, p)
    mAPs["cosim"] = calc_mAP(
        sim_matrix=sim_mat, 
        gallary_labels=gallary_labels, 
        query_labels=query_labels,
        embedding_gallary_name=embedding_gallary_name,
        verbose=verbose
    ) 
       
    ### EUCIDIAN DISTANCE AS A SIMULARITY METRIC ###
    if(verbose):
        cprint("\neuclidian distance", "cyan")
    file_name = embedding_gallary_name + "_sim_mat_eucl_dist.torch"
    p = dir / file_name
    if( p.exists() and not(exist_ok)):
        if(verbose):
            cprint("sim_mat already exists","green")
        sim_mat = torch.load(p)
    else:
        if(verbose):
            cprint("sim_mat doesn't exist yet or exist_ok=true, calulating...", "yellow")
        sim_mat = calc_eucl_dist_sim(query_stack=query_stack, embedding_gallary=embedding_gallary)
        torch.save(sim_mat, p)
    #reverse scores in simularity matrix for mAP calculation (low euclidian distance = high score and vice versa)
    sim_mat = sim_mat*-1
    mAPs["eucl_dist"] = calc_mAP(
        sim_matrix=sim_mat, 
        gallary_labels=gallary_labels, 
        query_labels=query_labels,
        embedding_gallary_name=embedding_gallary_name,
        verbose=verbose
    )
    
    ### EUCLIDIAN DISTANCE (NORMALIZED FEATURES) AS A SIMULARITY METRIC ###
    if(verbose):
        cprint("\neuclidian distance with normalized features", "cyan")
    file_name = embedding_gallary_name + "_sim_mat_eucl_dist_norm.torch"
    p = dir / file_name
    if( p.exists() and not(exist_ok)):
        if(verbose):
            cprint("sim_mat already exists","green")
        sim_mat = torch.load(p)
    else:
        if(verbose):
            cprint("sim_mat doesn't exist yet or exist_ok=true, calulating...", "yellow")
        sim_mat = calc_eucl_dist_sim(query_stack=query_stack_norm, embedding_gallary=embedding_gallary_norm)
        torch.save(sim_mat, p)
    #reverse scores in simularity matrix for mAP calculation (low euclidian distance = high score and vice versa)
    sim_mat = sim_mat*-1
    mAPs["eucl_dist_norm"] = calc_mAP(
        sim_matrix=sim_mat, 
        gallary_labels=gallary_labels, 
        query_labels=query_labels,
        embedding_gallary_name=embedding_gallary_name,
        verbose=verbose
    )
    
    if(verbose):
        print(mAPs)
        
    if(log_results):
        log_mAP_scores(dir, model_name, embedding_gallary_name, mAPs)      
    
def main():
    #choose an embedding gallary
    options = ["embedding_gallary", "embedding_gallary_avg"]
    print(f"Choose an embedding gallary to use. Your options are: {options}")
    gallary_name = input("Your choice: ")
    while gallary_name not in options:
        print(f"Invalid option. Your options are: {options}")
        gallary_name = input("Your Choice:") 
        
    #choose a model
    options = ["rotnet", "jigsaw", "moco32", "moco64", "simclr", "swav", "imgnet_pretrained", "all",
               "rotnet_phase0", "rotnet_phase25",  "rotnet_phase50", "rotnet_phase75","rotnet_phase100",
               "jigsaw_phase0", "jigsaw_phase25",  "jigsaw_phase50", "jigsaw_phase75","jigsaw_phase100",
               "moco32_phase0", "moco32_phase25",  "moco32_phase50", "moco32_phase75",
               "simclr_phase0", "simclr_phase25",  "simclr_phase50", "simclr_phase75"]
    print(f"Choose a model to calculate simularity matrices from with it's embedding gallary. Your options are: {options}")
    model_name = input("Your Choice:")
    while model_name not in options:
        print(f"Invalid option. Your options are: {options}")
        model_name = input("Your Choice:") 
    
    if(model_name == "all"):
        #Calculate simularity matrix for all models
        choice = input("At every checkpoint for all models? (y/N): ")
        if( choice != "y"):
            targets = ["rotnet", "jigsaw", "moco32", "simclr"]
        else:
            targets = ["rotnet", "rotnet_phase0", "rotnet_phase25",  "rotnet_phase50", "rotnet_phase75","rotnet_phase100", 
                       "jigsaw", "jigsaw_phase0", "jigsaw_phase25",  "jigsaw_phase50", "jigsaw_phase75","jigsaw_phase100",
                       "moco32", "moco32_phase0", "moco32_phase25",  "moco32_phase50", "moco32_phase75",
                       "simclr", "simclr_phase0", "simclr_phase25",  "simclr_phase50", "simclr_phase75" ]
        for target in targets:
            cprint(f" \nCalculating simularity matrix and mAP scores for model :{target}", "red")
            calc_sim_matrices(target, gallary_name,verbose=True, exist_ok=False, log_results=True)    
 
    else:
        calc_sim_matrices(model_name, gallary_name, verbose=True, exist_ok=True, log_results=True)
        
if __name__ == "__main__":
    main()
