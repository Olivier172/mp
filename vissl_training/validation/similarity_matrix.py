import torch 
import numpy as np
import os
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from termcolor import cprint
from sklearn.metrics import average_precision_score

from embedding_gallery import read_embedding_gallery

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
    
def calc_mAP(sim_matrix:torch.Tensor, gallery_labels:list , query_labels:list, verbose=False, mask_diagonal=False):
    """
    Function to calculate the mean Average Precision of a similarity matrix

    Args:
        sim_matrix (torch.Tensor): The similarity matrix to calculate the mAP of.
        gallery_labels (list): list of labels containing the class of every column in the similarity matrix 
                               (every class of the embedding gallery)
        query_labels (list): list of labels containing the ground truth classification label for every row in 
                             the similarity matrix (and each row contains the similarity scores for a query). 
        verbose (bool, optional): Boolean switch to enable prints. Defaults to False.
        mask_diagonal (bool, optional): Wether to mask confidence scores at the diagonal of sim_matrix for AP score calc. 
                                        This is necessary for the standard embedding gallery as there is a perfect match for every query. 
                                        We have to mask this value to not cosider this match in AP calculation. Defaults to False.

    Returns:
        mAP (float): The mAP score.
        amt_classes (int): amount of distinct classes in the labels of the gallery.
        amt_classes_AP (int): amount of distinct classes used for mAP calculation.
    """
    if(verbose):
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
          
    #dictionary to store average precisions for every query
    AP_queries = {}
    #calculate AP for every query in the sim_matrix
    for i in range(len(sim_matrix_np)):
        #y_true contains the ground truth for classification (True if label of query is the same)
        #gallery labels == query label 
        query_label = query_labels_np[i]
        y_true = gallery_labels_np == query_label #find all matches for the label of the current query (boolean np array)
        y_score = sim_matrix_np[i] 
        if(mask_diagonal):
            #in this case, the query embedding has a perfect match in the gallery
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
    
    amt_classes_AP_calc = len(AP_queries)
    if(verbose):
        print(f"amount of classes where an AP could be calculated {amt_classes_AP_calc}")
        
    #AP is average precision of a class with different threshods (positions in the PR curve)    
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
        
def generate_blacklist(gallery_labels:list, verbose=False):
    """
    Check how many occurences all the classes in gallery_labels have. Less then 3 means not eligible
    for use in embedding_gallery_avg, so they are added to the blacklist.
    class occurences will be saved to "data/class_occurences.json".
    blacklist will be saved to data/blacklist.txt".

    Args:
        gallery_labels (list): list of all the labels in the embedding_gallery.
        verbose (bool, optional): Controls prints.
        
    Returns:
        blacklist (list): a list containing all the class names that are not eligable for use in embedding_gallery_avg.
    """
    if(verbose):
        cprint("in function generate_blacklist()", "green")
    #Lets count how many times each label occurs
    counter = Counter(gallery_labels)
    #list to store all elements that occur only once
    blacklist = list()
    #store the amount of occurences per class
    class_occurences = {} 
    for item, count in counter.items():
        class_occurences[item] = count
        if(count < 3):
            blacklist.append(item)
            
    if(verbose):
        cprint(f"Info: {len(blacklist)} items were added to the blacklist", "red")
                
    #save blacklist to output file
    blacklist_file = Path("data/blacklist.txt")
    if(not(blacklist_file.is_file())):
        with open(blacklist_file, "w") as f:
            cprint(f"Saving blacklist to file {blacklist_file}", "green")
            f.writelines("\n".join(blacklist))
        
    #save class occurences 
    occurence_json_file = Path("data/class_occurences.json")
    if(not(occurence_json_file.is_file())):
        with open(occurence_json_file, "w") as f:
            cprint(f"Saving class occurences to file {occurence_json_file}", "green")
            json.dump(class_occurences, f)
            
    return blacklist

def calc_embedding_gallery_avg(
    output_file:Path, 
    dir:Path, 
    model_name:str, 
    embedding_gallery:torch.Tensor, 
    embedding_gallery_norm:torch.Tensor, 
    gallery_labels:list, 
    verbose=False, 
    exist_ok=False
):
    """
    Calculating average embedding gallery based on the standard embedding gallery.

    Args:
        output_file (Path): file to log results to.
        dir (Path): folder to log results to.
        model_name (str): name of model used to calc embedding_gallery.
        embedding_gallery (torch.Tensor): standard embedding gallery.
        embedding_gallery_norm (torch.Tensor): normalized standard embedding gallery.
        gallery_labels (list): ground truth labels for embeddings in embedding_gallery
        verbose (bool, optional): switch to control prints. Defaults to False.
    """
    if(verbose):
        print("In function calc_embedding_gallery_avg", "yellow")
    #Check if calc is necessary
    if(output_file.is_file() and not(exist_ok)):
        cprint(f"Info: {output_file} aready exists, skip calculation", "yellow")
        return
    
    if(verbose):
        cprint("Calculating an embedding_gallery_avg", "red")
        
    #get blacklist to exclude all uneligible for the caculation of the avg embedding gallery
    blacklist = generate_blacklist(gallery_labels=gallery_labels, verbose=verbose)
    #dictionary to summarize embedding gallery per class
    class_embeddings = {}
    #collect all embeddings per class and list them up in the dict:
    for idx, embedding in enumerate(embedding_gallery):
        class_name = gallery_labels[idx]
        #skip uneligible classes
        if(class_name in  blacklist):
            continue
        if(class_name not in class_embeddings.keys()):
            #register class in dict
            class_embeddings[class_name]=[]
        #add embedding to the list of embeddings of this class:
        class_embeddings[class_name].append(embedding)
    
    #calculate the embedding_gallery_avg and query stack
    class_labels = list(class_embeddings.keys())
    if(verbose):
        print(f"Calculating avg embedding gallery with {len(class_labels)} classes")
    #gallery
    embedding_gallery_avg = []
    gallery_avg_labels = []
    #query stack (test set)
    query_stack = []
    query_labels = []
    for class_name in class_labels:
        #first embedding of the class is the test query for this class
        query = class_embeddings[class_name][0]
        query_stack.append(query)
        query_labels.append(class_name)
        
        #caculate the avg embedding of this class with the remaining embeddings
        embeddings = class_embeddings[class_name][1:] #remaining embeddings
        embedding_avg = torch.mean(torch.stack(embeddings), dim=0) #averaging the tensors (embeddings) of this class
        embedding_gallery_avg.append(embedding_avg)
        gallery_avg_labels.append(class_name)
    #convert to tensors
    embedding_gallery_avg = torch.stack(embedding_gallery_avg)
    embedding_gallery_avg_norm = embedding_gallery_avg / embedding_gallery_avg.norm(dim=1,keepdim=True) 
    query_stack = torch.stack(query_stack)
    query_stack_norm = query_stack / query_stack.norm(dim=1,keepdim=True) 
    
    
    #Dictionary to save mAP scores for each metric
    mAPs = {
        "ip": 0.0, "cosim": 0.0, "eucl_dist": 0.0, "eucl_dist_norm": 0.0 
    }
    
    ### INNER PRODUCT AS A similarity METRIC ###
    if(verbose):
        cprint("\ninner product", "cyan")
        
    sim_mat = calc_ip_cosim(
        query_stack=query_stack, 
        embedding_gallery=embedding_gallery_avg, 
        verbose=verbose
    ) 
    mAPs["ip"], amt_classes, amt_classes_AP_calc = calc_mAP(
        sim_matrix=sim_mat, 
        gallery_labels=gallery_avg_labels, 
        query_labels=query_labels, 
        verbose=verbose,
        mask_diagonal=False
    ) 
    
    ### COSINE similarity AS A similarity METRIC ###
    if(verbose):
        cprint("\ncosine similarity", "cyan")

    sim_mat = calc_ip_cosim(
        query_stack=query_stack_norm, 
        embedding_gallery=embedding_gallery_avg_norm, 
        verbose=verbose
    )   
    mAPs["cosim"], amt_classes, amt_classes_AP_calc = calc_mAP(
        sim_matrix=sim_mat, 
        gallery_labels=gallery_avg_labels, 
        query_labels=query_labels,
        verbose=verbose,
        mask_diagonal=False
    ) 
       
    ### EUCIDIAN DISTANCE AS A similarity METRIC ###
    if(verbose):
        cprint("\neuclidian distance", "cyan")
        
    sim_mat = calc_eucl_dist_sim(
        query_stack=query_stack, 
        embedding_gallery=embedding_gallery_avg, 
        verbose=verbose
    )   
    #reverse scores in similarity matrix for mAP calculation (low euclidian distance = high score and vice versa)
    sim_mat = sim_mat*-1
    mAPs["eucl_dist"], amt_classes, amt_classes_AP_calc = calc_mAP(
        sim_matrix=sim_mat, 
        gallery_labels=gallery_avg_labels, 
        query_labels=query_labels,
        verbose=verbose,
        mask_diagonal=False
    )
    
    ### EUCLIDIAN DISTANCE (NORMALIZED FEATURES) AS A similarity METRIC ###
    if(verbose):
        cprint("\neuclidian distance with normalized features", "cyan")

    sim_mat = calc_eucl_dist_sim(
        query_stack=query_stack_norm, 
        embedding_gallery=embedding_gallery_avg_norm, 
        verbose=verbose
    )
    #reverse scores in similarity matrix for mAP calculation (low euclidian distance = high score and vice versa)
    sim_mat = sim_mat*-1
    mAPs["eucl_dist_norm"], amt_classes, amt_classes_AP_calc= calc_mAP(
        sim_matrix=sim_mat, 
        gallery_labels=gallery_avg_labels, 
        query_labels=query_labels,
        verbose=verbose,
        mask_diagonal=False
    )
    mAPs["amt_classes"] = amt_classes
    mAPs["amt_classes_AP_calc"] = amt_classes_AP_calc
    
    if(verbose):
        print(mAPs)
        
    #log results standard embedding gallery
    output_file = dir / "embedding_gallery_avg_mAP_scores.txt"
    msg = f"avg embedding gallery calculated with {len(class_labels)} classes\n"
    log_mAP_scores(output_file=output_file, model_name=model_name, mAPs=mAPs, optional_trunk=msg) 
    
    
def calc_sim_matrices(model_name:str, verbose=False, exist_ok=False, log_results=False):
    """
    Calulates the similarity matrices of the entire embedding gallery.
    This is done using 4 different metrics of similarity. (Generating 4 similarity matrices) 
    A mean Average Precision score is also calculated using the similarity matrix.
    
    Args:
        model_name (str): The name of the model to calculate the similarity matrix for based on the 
                          embedding gallery for this model.
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
    embedding_gallery, embedding_gallery_norm, gallery_labels = read_embedding_gallery(dir, verbose=verbose)
    
    #Acquiring query stack
    query_stack = embedding_gallery.clone().detach()
    query_stack_norm = embedding_gallery_norm.clone().detach()
    query_labels = gallery_labels.copy()
    
    #Dictionary to save mAP scores for each metric
    mAPs = {
        "ip": 0.0, "cosim": 0.0, "eucl_dist": 0.0, "eucl_dist_norm": 0.0 
    }
    
    ### INNER PRODUCT AS A similarity METRIC ###
    if(verbose):
        cprint("\ninner product", "cyan")
    p = dir / "embedding_gallery_sim_mat_ip.torch"
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
        verbose=verbose,
        mask_diagonal=True
    ) 
    
    ### COSINE similarity AS A similarity METRIC ###
    if(verbose):
        cprint("\ncosine similarity", "cyan")
    p = dir / "embedding_gallery_sim_mat_cosim.torch"
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
        verbose=verbose,
        mask_diagonal=True
    ) 
       
    ### EUCIDIAN DISTANCE AS A similarity METRIC ###
    if(verbose):
        cprint("\neuclidian distance", "cyan")
    p = dir / "embedding_gallery_sim_mat_eucl_dist.torch"
    if( p.exists() and not(exist_ok)):
        if(verbose):
            cprint("Info: sim_mat already exists","green")
        sim_mat = torch.load(p)
    else:
        if(verbose):
            cprint("Info: sim_mat doesn't exist yet or exist_ok=true, calulating...", "yellow")
        sim_mat = calc_eucl_dist_sim(query_stack=query_stack, embedding_gallery=embedding_gallery, verbose=verbose)
        torch.save(sim_mat, p)
        
    #reverse scores in similarity matrix for mAP calculation (low euclidian distance = high score and vice versa)
    sim_mat = sim_mat*-1
    mAPs["eucl_dist"], amt_classes, amt_classes_AP_calc = calc_mAP(
        sim_matrix=sim_mat, 
        gallery_labels=gallery_labels, 
        query_labels=query_labels,
        verbose=verbose,
        mask_diagonal=True
    )
    
    ### EUCLIDIAN DISTANCE (NORMALIZED FEATURES) AS A similarity METRIC ###
    if(verbose):
        cprint("\neuclidian distance with normalized features", "cyan")
    p = dir / "embedding_gallery_sim_mat_eucl_dist_norm.torch"
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
        verbose=verbose,
        mask_diagonal=True
    )
    mAPs["amt_classes"] = amt_classes
    mAPs["amt_classes_AP_calc"] = amt_classes_AP_calc
    
    if(verbose):
        print(mAPs)
        
    if(log_results):
        #log results standard embedding gallery
        output_file = dir / "embedding_gallery_mAP_scores.txt"
        if(verbose):
            cprint(f"Logging mAP scores from query matching to {output_file}","green")
        log_mAP_scores(output_file=output_file, model_name=model_name, mAPs=mAPs)  
        #Calc and log results avg embedding gallery
        output_file_avg = dir / "embedding_gallery_avg_mAP_scores.txt"
        calc_embedding_gallery_avg(
            output_file=output_file_avg,
            dir=dir,
            model_name=model_name,
            embedding_gallery=embedding_gallery,
            embedding_gallery_norm=embedding_gallery_norm,
            gallery_labels=gallery_labels,
            verbose=verbose,
            exist_ok=exist_ok
        )
        
            
def log_mAP_scores(output_file:Path, model_name:str, mAPs:dict, optional_trunk:str=None):
    """
    Function to log mAP scores to a file. Results are appended to dir/mAP_scores.txt
    
    Args:
        output_file (Path): path where the logfile should be stored.
        model_name (str): name of the model used.
        mAPs: dictionary with mAP scores.
        optional_trunk (str, optional): extra string to print at the end of log.
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
    lines.append(f"Embedding gallery of model {model_name} was used to calculated these scores.\n")
    lines.append(f"In total there where {mAPs['amt_classes']} distinct classes in the test set of which {mAPs['amt_classes_AP_calc']} classes could be used for mAP calcuation.\n")
    lines.append(f"mAP={mAPs['ip']*100}% for inner product (ip) as a similarity metric.\n")
    lines.append(f"mAP={mAPs['cosim']*100}% for cosine similarity (cosim) as a similarity metric.\n")
    lines.append(f"mAP={mAPs['eucl_dist']*100}% for euclidian distance (eucl_dist) as a similarity metric.\n")
    lines.append(f"mAP={mAPs['eucl_dist_norm']*100}% for euclidian distance with normalized features (eucl_dist_norm) as a similarity metric.\n")
    if(optional_trunk != None):
        lines.append(optional_trunk)
    lines.append("-"*90 + "\n\n")
    
    #Saving to file
    cprint(f"Logging mAP scores to a file on path : {output_file}", "green")
    with open(output_file, "w") as f:
        f.writelines(lines)
 
def compile_total_log(output_file:Path, dir:Path, targets:list, file_names:list):
    """
    Compiles a total log that consists of smaller logs per target. Log is saved to output_file.

    Args:
        output_file (Path): file to save total log to.
        dir (Path): folder to where all the targets are saved. (So that small logs can be collected there)
        targets (list<str>): list of models to look for small logs and collect and include them in the total log.
        file_names (list<str>): log file names that are collected per model.
    """
    output = []
    # Get the current date and time
    current_datetime = datetime.now()
    # Extract the date and time components
    current_date = current_datetime.date()
    current_time = current_datetime.time()
    # Convert to string format
    date_string = current_date.strftime('%Y-%m-%d')
    time_string = current_time.strftime('%H:%M:%S')
    output.append(f"Total log compiled at {date_string} @ {time_string}.\n")
    #gather all seperate logs
    for target in targets:
        output.append(f"\nResults for {target}\n")
        for file in file_names:
            p:Path
            p = dir / target / file
            if(p.is_file()):
                output.append(f"origin_file: {p}\n")
                with open(p, "r") as f:
                    data = f.readlines()
                    output.extend(data)
                    output.append("\n")
            else:
                cprint(f"Warning file {p} not found to compile total log and will not be included", "red")
                
    #write the compiled version to a file
    cprint(f"Logging compiled log to {output_file}","green")
    with open(output_file, "w") as f:
        f.writelines(output)       
        
def get_targets(prompt:str="Choose a model"):
    """
    User input function to get all target models.
    
    Args:
        prompt(str, optional): the prompt to display when choosing models
    
    Returns:
        targets(list<str>): a list of model names to use
    """
    #choose a model
    options = ["rotnet", "jigsaw", "moco32", "moco64", "simclr", "swav", "imgnet_pretrained", "random","all",
               "rotnet_phase0", "rotnet_phase25",  "rotnet_phase50", "rotnet_phase75","rotnet_phase100",
               "jigsaw_phase0", "jigsaw_phase25",  "jigsaw_phase50", "jigsaw_phase75","jigsaw_phase100",
               "moco32_phase0", "moco32_phase25",  "moco32_phase50", "moco32_phase75",
               "moco64_phase0", "moco64_phase25",  "moco64_phase50", "moco64_phase75",
               "simclr_phase0", "simclr_phase25",  "simclr_phase50", "simclr_phase75",
               "swav_phase0", "swav_phase25",  "swav_phase50", "swav_phase75"]
    
    print(f"{prompt}. Your options are: {options}")
    model_name = input("Your Choice:")
    while model_name not in options:
        print(f"Invalid option. Your options are: {options}")
        model_name = input("Your Choice:")
        
    if(model_name == "all"):
        #all models
        choice = input("At every checkpoint for all models? (y/N): ")
        if( choice != "y"):
            targets = ["rotnet", "jigsaw", "moco32", "moco64", "simclr", "swav", "imgnet_pretrained", "random"]
        else:
            targets = ["rotnet", "rotnet_phase0", "rotnet_phase25",  "rotnet_phase50", "rotnet_phase75","rotnet_phase100", 
                       "jigsaw", "jigsaw_phase0", "jigsaw_phase25",  "jigsaw_phase50", "jigsaw_phase75","jigsaw_phase100",
                       "moco32", "moco32_phase0", "moco32_phase25",  "moco32_phase50", "moco32_phase75",
                       "moco64", "moco64_phase0", "moco64_phase25",  "moco64_phase50", "moco64_phase75",
                       "simclr", "simclr_phase0", "simclr_phase25",  "simclr_phase50", "simclr_phase75",
                       "swav", "swav_phase0", "swav_phase25",  "swav_phase50", "swav_phase75",
                       "imgnet_pretrained", "random"]
    else:
        #one model
        targets = [model_name]
    return targets

def main():
        
    #choose a model
    targets = get_targets("Choose a model to calculate similarity matrices from with it's embedding gallery")
    for target in targets:
        cprint(f" \nCalculating similarity matrix and mAP scores for model :{target}", "red")
        calc_sim_matrices(
            target, 
            verbose=True, 
            exist_ok=False, 
            log_results=True
        ) 
    
    #dir to collect logs per target
    dir = Path("data")
    f_name = f"total_gallery_mAP_scores_log.txt"
    #files of logs to collect
    file_names = ["embedding_gallery_mAP_scores.txt", "embedding_gallery_avg_mAP_scores.txt"]
    #file to save log to
    output_file = dir / f_name
    compile_total_log(
        output_file=output_file,
        dir=dir,
        targets=targets,
        file_names = file_names
    )   
        
if __name__ == "__main__":
    main()
