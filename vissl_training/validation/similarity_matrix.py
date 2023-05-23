import torch 
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from termcolor import cprint
from sklearn.metrics import average_precision_score

def read_embedding_gallery(dir:Path, embedding_gallery_name:str):
    """
    Reads in the embedding gallery from disk.

    Args:
        dir (Path): Path to the directory where the gallery is saved.
        embedding_gallery_name (str): The name of the embedding gallery used. 
                                      This can be "embedding_gallery" or "embedding_gallery_avg" 

    Returns:
        embedding_gallery: The embedding gallery contains a stack of embeddings for which the label is known.
        embedding_gallery_norm: gallery with normalized embeddings.
        labels: Ground turth labels for every row (embedding) in the embedding gallery.
    """
    cprint("In function read_embedding_gallery()", "green")
    
    file_name = embedding_gallery_name + ".torch" 
    embedding_gallery = torch.load(dir / file_name)
    print(f"fts_stack has shape {embedding_gallery.shape}")
    
    file_name = embedding_gallery_name + "_norm.torch"
    embedding_gallery_norm = torch.load(dir / file_name)
    print(f"fts_stack_norm has shape {embedding_gallery_norm.shape}")
    labels = list()
    
    file_name = embedding_gallery_name + "_labels.txt"
    with open(dir / file_name, "r") as f:
        labels = f.read().splitlines()
        print(f"labels list has length "+ str(len(labels)))
     
    return embedding_gallery, embedding_gallery_norm, labels

def read_blacklist(file_path:Path) -> list:
    """
    Reads in the blacklist for class labels not to use when evaluating mAP scores of embedding_gallery_avg.
    Because these classes contain only one element in the CornerShop dataset and would be an "easy match".

    Args:
        file_path (Path): path to the blacklist file.

    Returns:
        blacklist (list): list of strings containing the blacklisted classes. 
    """
    if(not os.path.isfile):
        cprint(f"Warning: blacklist file for embedding_gallery_avg doesn't exist at {file_path}", "red")
        return []
    
    with open(file_path, "r") as f:
        blacklist = f.read().splitlines()
        cprint(f"Blacklist file succesfully read. There are  "+ str(len(blacklist)) + " classes blacklisted.", "green")
    return blacklist

def calc_ip_cosim(query_stack:torch.Tensor, embedding_gallery:torch.Tensor, verbose=False):
    """
    Calulates a similarity matrix based on the inner product / cosine similarity as a metric.
        -If you use normalized features/embeddings for the query_stack and embedding gallery, the cosine similarity (cosim) matrix 
        is calculated.
        -If you use unnormalized features/embeddings for the query_stack and embedding gallery, the inner product (ip) matrix 
        is calculated.
        
    Args:
        query_stack (torch.Tensor): This is a stack of all query embeddings to match to the embedding gallery.
        embedding_gallery (torch.Tensor): This is the embedding gallery itself. Each row consist of one embedding
        for which the label (ground truth) is known. The query embeddings are matched to these gallery embeddings.

    Returns:
        sim_matrix (torch.Tensor): The similarity matrix. This matrix has in each row similarity scores for one query
        from the query stack to all embeddings in the embedding gallery.
    """
    #Calculate similarity from each embedding of the query_stack with every embedding from the embedding gallery by doing a matrix product:
    sim_matrix = query_stack.matmul(embedding_gallery.T)
    if(verbose):
        #statistics
        print(f"Max: {sim_matrix.max()}")
        print(f"Min: {sim_matrix.min()}")
        print(f"mean: {torch.mean(sim_matrix)}")
        print(f"std: {torch.std(sim_matrix)}")
    return sim_matrix

def calc_eucl_dist_sim(query_stack:torch.Tensor, embedding_gallery:torch.Tensor, verbose=False) -> torch.Tensor:
    """
    Calulates a similarity matrix based on the euclidian distance as a metric.

    Args:
        query_stack (torch.Tensor): This is a stack of all query embeddings to match to the embedding gallery.
        embedding_gallery (torch.Tensor): This is the embedding gallery itself. Each row consist of one embedding
        for which the label (ground truth) is known. The query embeddings are matched to these gallery embeddings.

    Returns:
        sim_matrix (torch.Tensor): The similarity matrix. This matrix has in each row similarity scores for one query
        from the query stack to all embeddings in the embedding gallery.
    """
    eucl_dists = [] 
    for tensor in query_stack:
        d = [] #store all distances from this tensor to all the other tensors
        for other_tensor in embedding_gallery:
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
    
def calc_mAP(sim_matrix:torch.Tensor, gallery_labels:list , query_labels:list, embedding_gallery_name:str, verbose=False):
    """
    Function to calculate the mean Average Precision of a similarity matrix

    Args:
        sim_matrix (torch.Tensor): The similarity matrix to calculate the mAP of.
        gallery_labels (list): list of labels containing the class of every column in the similarity matrix 
                               (every class of the embedding gallery)
        query_labels (list): list of labels containing the ground truth classification label for every row in 
                             the similarity matrix (and each row contains the similarity scores for a query). 
        verbose (bool, optional): Boolean switch to enable prints. Defaults to False.

    Returns:
        mAP (float): The mAP score.
        amt_classes (int): amount of distinct classes in the labels of the gallery.
        amt_classes_AP (int): amount of distinct classes used for mAP calculation.
    """
    cprint("In function calc_mAP()", "green")
    #Convert everything to numpy
    sim_matrix_np = sim_matrix.numpy()
    gallery_labels_np = np.array(gallery_labels)
    query_labels_np = np.array(query_labels)
    if(verbose):
        print(f"sim_matrix_np shape {sim_matrix_np.shape}")
        print(f"gallery_labels shape {gallery_labels_np.shape}")
        print(f"query_labels shape {query_labels_np.shape}")
        print(f"len sim_matrix_np {len(sim_matrix_np)}")
    
    
    #return indicis of all unique class labels
    classes = list(np.unique(gallery_labels_np))
    amt_classes = len(classes)
    if(verbose):
        print(f"amount of classes {amt_classes}")
      
    #Read in blacklist for fair evaluation of embedding_gallery_avg
    if(embedding_gallery_name == "embedding_gallery_avg"):
          blacklist = read_blacklist(Path("data/blacklist.txt"))  
          
    #dictionary to store average precisions for every query
    AP_queries = {}
    #calculate AP for every query in the sim_matrix
    for i in range(len(sim_matrix_np)):
        #y_true contains the ground truth for classification (True if label of query is the same)
        #gallery labels == query label 
        query_label = query_labels_np[i]
        y_true = gallery_labels_np == query_label #find all matches for the label of the current query (boolean np array)
        y_score = sim_matrix_np[i] 
        if(embedding_gallery_name == "embedding_gallery"):
            #in this case, the query embedding has a perfect match in the gallery
            #we need to remove this one before evaluating AP to prevent "easy match"
            y_true[i] = False #Remove perfect match
            y_score[i]= -99999.0 #large negative number to indicate no match
        elif(embedding_gallery_name == "embedding_gallery_avg"):
            #skip queries of blacklisted classes for embedding_gallery_avg to prevent easy matches
            if(query_label in blacklist):
                continue
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
    
    amt_classes_AP_calc = len(AP_queries)
    if(verbose):
        print(f"amount of classes where an AP could be calculated {amt_classes_AP_calc}")
        
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
        
    return mAP, amt_classes, amt_classes_AP_calc
    
def log_mAP_scores(dir: Path, model_name:str, embedding_gallery_name:str, mAPs:dict):
    """
    Function to log mAP scores to a file. Results are appended to dir/mAP_scores.txt
    
    Args:
        dir (Path): directory to place logfile.
        model_name (str): name of the model used.
        embedding_gallery_name (str): The name of the embedding gallery used. 
                                      This can be "embedding_gallery" or "embedding_gallery_avg" 
        mAPs: dictionary with mAP scores.
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
    lines.append(f"mAP Logging results on {date_string} @ {time_string}.\n")
    lines.append(f"Embedding library of model {model_name} was used to calculated these scores.\n")
    lines.append(f"In total there where {mAPs['amt_classes']} distinct classes in the test set of which {mAPs['amt_classes_AP_calc']} classes could be used for mAP calcuation.\n")
    lines.append(f"mAP={mAPs['ip']} for inner product (ip) as a similarity metric.\n")
    lines.append(f"mAP={mAPs['cosim']} for cosine similarity (cosim) as a similarity metric.\n")
    lines.append(f"mAP={mAPs['eucl_dist']} for euclidian distance (eucl_dist) as a similarity metric.\n")
    lines.append(f"mAP={mAPs['eucl_dist_norm']} for euclidian distance with normalized features (eucl_dist_norm) as a similarity metric.\n")
    lines.append("-"*90 + "\n\n")
    
    #Saving to file
    file_name = embedding_gallery_name + "_mAP_scores.txt"
    p = dir / file_name
    cprint(f"Logging mAP scores to a file on path : {p}", "green")
    with open(p, "a") as f:
        f.writelines(lines)
        
def calc_sim_matrices(model_name:str, embedding_gallery_name:str, verbose=False, exist_ok=False, log_results=False):
    """
    Calulates the similarity matrices of the entire embedding gallery.
    This is done using 4 different metrics of similarity. (Generating 4 similarity matrices) 
    A mean Average Precision score is also calculated using the similarity matrix.
    
    Args:
        model_name (str): The name of the model to calculate the similarity matrix for based on the 
                          embedding gallery for this model.
        embedding_gallery_name (str): The name of the embedding gallery used. 
                                      This can be "embedding_gallery" or "embedding_gallery_avg" 
        exist_ok (Bool): Boolean switch to recalculate and overwrite sim_matrix if true.
        verbose (Bool): Boolean switch to allow prints of this function.
        log_results (Bool): Boolean switch to log mAP scores to a logfile.
    """
    if(verbose):
        cprint("In function calc_sim_matrix()", "green")
    embedding_gallery:torch.Tensor
    embedding_gallery_norm:torch.Tensor
    gallery_labels:list
    query_labels:list
    
    #Reading in the embedding gallery
    dir = Path("data/" + model_name) #Directory to find gallaries for this model
    embedding_gallery, embedding_gallery_norm, gallery_labels = read_embedding_gallery(dir, embedding_gallery_name)
    
    #Reading in the query stack
    if(embedding_gallery_name == "embedding_gallery"):
        #in this case we query all the embeddings from the embedding gallery itself as a test set (the entire cornershop dataset embeddings)
        query_stack = embedding_gallery.clone().detach()
        query_stack_norm = embedding_gallery_norm.clone().detach()
        query_labels = gallery_labels.copy()
    else: #embedding_gallery_name == embedding_gallery_avg
        #in this case we query all the embeddings from the cornershop dataset (which are stored in the standard embedding gallery)
        #but the embedding gallery itself contains different embeddings (averages per class)
        query_stack, query_stack_norm, query_labels = read_embedding_gallery(dir, "embedding_gallery")
        
    #Dictionary to save mAP scores for each metric
    mAPs = {
        "ip": 0.0, "cosim": 0.0, "eucl_dist": 0.0, "eucl_dist_norm": 0.0 
    }
    
    ### INNER PRODUCT AS A similarity METRIC ###
    if(verbose):
        cprint("\ninner product", "cyan")
    file_name = embedding_gallery_name + "_sim_mat_ip.torch"
    p = dir / file_name
    if( p.exists() and not(exist_ok)):
        if(verbose):
            cprint("sim_mat already exists","green")
        sim_mat = torch.load(p)
    else:
        if(verbose):
            cprint("sim_mat doesn't exist yet or exist_ok=true, calulating...", "yellow")
        sim_mat = calc_ip_cosim(query_stack=query_stack, embedding_gallery=embedding_gallery, verbose=verbose)
        torch.save(sim_mat, p)
    mAPs["ip"], amt_classes, amt_classes_AP_calc = calc_mAP(
        sim_matrix=sim_mat, 
        gallery_labels=gallery_labels, 
        query_labels=query_labels, 
        embedding_gallery_name=embedding_gallery_name,
        verbose=verbose
    ) 
    
    ### COSINE similarity AS A similarity METRIC ###
    if(verbose):
        cprint("\ncosine similarity", "cyan")
    file_name = embedding_gallery_name + "_sim_mat_cosim.torch"
    p = dir / file_name
    if( p.exists() and not(exist_ok)):
        if(verbose):
            cprint("sim_mat already exists","green")
        sim_mat = torch.load(p)
    else:
        if(verbose):
            cprint("sim_mat doesn't exist yet or exist_ok=true, calulating...", "yellow")
        sim_mat = calc_ip_cosim(query_stack=query_stack_norm, embedding_gallery=embedding_gallery_norm, verbose=verbose)
        torch.save(sim_mat, p)
    mAPs["cosim"], amt_classes, amt_classes_AP_calc = calc_mAP(
        sim_matrix=sim_mat, 
        gallery_labels=gallery_labels, 
        query_labels=query_labels,
        embedding_gallery_name=embedding_gallery_name,
        verbose=verbose
    ) 
       
    ### EUCIDIAN DISTANCE AS A similarity METRIC ###
    if(verbose):
        cprint("\neuclidian distance", "cyan")
    file_name = embedding_gallery_name + "_sim_mat_eucl_dist.torch"
    p = dir / file_name
    if( p.exists() and not(exist_ok)):
        if(verbose):
            cprint("sim_mat already exists","green")
        sim_mat = torch.load(p)
    else:
        if(verbose):
            cprint("sim_mat doesn't exist yet or exist_ok=true, calulating...", "yellow")
        sim_mat = calc_eucl_dist_sim(query_stack=query_stack, embedding_gallery=embedding_gallery, verbose=verbose)
        torch.save(sim_mat, p)
    #reverse scores in similarity matrix for mAP calculation (low euclidian distance = high score and vice versa)
    sim_mat = sim_mat*-1
    mAPs["eucl_dist"], amt_classes, amt_classes_AP_calc = calc_mAP(
        sim_matrix=sim_mat, 
        gallery_labels=gallery_labels, 
        query_labels=query_labels,
        embedding_gallery_name=embedding_gallery_name,
        verbose=verbose
    )
    
    ### EUCLIDIAN DISTANCE (NORMALIZED FEATURES) AS A similarity METRIC ###
    if(verbose):
        cprint("\neuclidian distance with normalized features", "cyan")
    file_name = embedding_gallery_name + "_sim_mat_eucl_dist_norm.torch"
    p = dir / file_name
    if( p.exists() and not(exist_ok)):
        if(verbose):
            cprint("sim_mat already exists","green")
        sim_mat = torch.load(p)
    else:
        if(verbose):
            cprint("sim_mat doesn't exist yet or exist_ok=true, calulating...", "yellow")
        sim_mat = calc_eucl_dist_sim(query_stack=query_stack_norm, embedding_gallery=embedding_gallery_norm, verbose=verbose)
        torch.save(sim_mat, p)
    #reverse scores in similarity matrix for mAP calculation (low euclidian distance = high score and vice versa)
    sim_mat = sim_mat*-1
    mAPs["eucl_dist_norm"], amt_classes, amt_classes_AP_calc= calc_mAP(
        sim_matrix=sim_mat, 
        gallery_labels=gallery_labels, 
        query_labels=query_labels,
        embedding_gallery_name=embedding_gallery_name,
        verbose=verbose
    )
    mAPs["amt_classes"] = amt_classes
    mAPs["amt_classes_AP_calc"] = amt_classes_AP_calc
    
    if(verbose):
        print(mAPs)
        
    if(log_results):
        log_mAP_scores(dir, model_name, embedding_gallery_name, mAPs)      
    
def main():
    #choose an embedding gallery
    options = ["embedding_gallery", "embedding_gallery_avg"]
    print(f"Choose an embedding gallery to use. Your options are: {options}")
    gallery_name = input("Your choice: ")
    while gallery_name not in options:
        print(f"Invalid option. Your options are: {options}")
        gallery_name = input("Your Choice:") 
        
    #choose a model
    options = ["rotnet", "jigsaw", "moco32", "moco64", "simclr", "swav", "imgnet_pretrained", "all",
               "rotnet_phase0", "rotnet_phase25",  "rotnet_phase50", "rotnet_phase75","rotnet_phase100",
               "jigsaw_phase0", "jigsaw_phase25",  "jigsaw_phase50", "jigsaw_phase75","jigsaw_phase100",
               "moco32_phase0", "moco32_phase25",  "moco32_phase50", "moco32_phase75",
               "moco64_phase0", "moco64_phase25",  "moco64_phase50", "moco64_phase75",
               "simclr_phase0", "simclr_phase25",  "simclr_phase50", "simclr_phase75"]
    print(f"Choose a model to calculate similarity matrices from with it's embedding gallery. Your options are: {options}")
    model_name = input("Your Choice:")
    while model_name not in options:
        print(f"Invalid option. Your options are: {options}")
        model_name = input("Your Choice:") 
    
    if(model_name == "all"):
        #Calculate similarity matrix for all models
        choice = input("At every checkpoint for all models? (y/N): ")
        if( choice != "y"):
            targets = ["rotnet", "jigsaw", "moco32", "moco64", "simclr", "imgnet_pretrained"]
        else:
            targets = ["rotnet", "rotnet_phase0", "rotnet_phase25",  "rotnet_phase50", "rotnet_phase75","rotnet_phase100", 
                       "jigsaw", "jigsaw_phase0", "jigsaw_phase25",  "jigsaw_phase50", "jigsaw_phase75","jigsaw_phase100",
                       "moco32", "moco32_phase0", "moco32_phase25",  "moco32_phase50", "moco32_phase75",
                       "moco64", "moco64_phase0", "moco64_phase25",  "moco64_phase50", "moco64_phase75",
                       "simclr", "simclr_phase0", "simclr_phase25",  "simclr_phase50", "simclr_phase75",
                       "imgnet_pretrained"]
        for target in targets:
            cprint(f" \nCalculating similarity matrix and mAP scores for model :{target}", "red")
            calc_sim_matrices(target, gallery_name,verbose=True, exist_ok=False, log_results=True)    
 
    else:
        calc_sim_matrices(model_name, gallery_name, verbose=True, exist_ok=False, log_results=True)
        
if __name__ == "__main__":
    main()
