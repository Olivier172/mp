import torch 
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from termcolor import cprint
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve 

def read_embedding_gallary(dir:Path):
    cprint("In function calc_sim_matrix()", "green")
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
    """Calulates a simularity matrix based on the inner product / cosine simularity a metric.
    Args:
        fts_stack (torch.Tensor): fts_stack: stack of embeddings -> inner product
        fts_stack_norm: stack of normalized embeddings -> cosim

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
    """Calulates a simularity matrix based on the euclidian distance as a metric.

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
    
def calc_avg_precision(sim_matrix:torch.Tensor, labels, verbose=False):
    cprint("In function calc_avg_precision", "green")
    sim_matrix_np = sim_matrix.numpy()
    labels_np = np.array(labels)
    if(verbose):
        print(f"sim_matrix_np shape {sim_matrix_np.shape}")
        print(f"labels shape {labels_np.shape}")
        print(f"len sim_matrix_np {len(sim_matrix_np)}")
    
    
    #return indicis of all unique class labels
    #class_idxs = np.unique(labels_np, return_index=True)
    #calculate mAP, mean over all AP for each class
    #AP is average precision of a class with different threshods (poisitions in the PR curve)
    mAP = 0
    for i in range(len(sim_matrix_np)):
        #y_true contains the ground truth for classification (True if label of query is the same)
        #gallary labels == query label (query is on the diagonal so the i th element of the i th row)
        y_true = labels_np == labels_np[i] #find all matches for the label of the current query (boolean np array)
        y_score = sim_matrix_np[i] 
        #compute AP
        AP = average_precision_score(y_true=y_true, y_score=y_score)
        mAP += AP 
        
    mAP /= len(sim_matrix_np)
    if(verbose):
        cprint(f"mAP of this model is: {mAP}","magenta")
    
        
def calc_sim_matrix(model_name:str):
    """Calulates the simularity matrix of the entire embedding gallary
    """
    fts_stack:torch.Tensor
    fts_stack_norm:torch.Tensor
    #read input gallary for this model
    dir = Path("data/" + model_name)
    fts_stack, fts_stack_norm, labels = read_embedding_gallary(dir)
    #transform to np arrays
    #fts_stack_np, fts_stack_norm_np = fts_stack.numpy(), fts_stack_norm.numpy()
    
    
    # cprint("inner product", "cyan")
    # sim_mat = calc_eucl_dist_sim(fts_stack=fts_stack)
    
    cprint("cosine simularity", "cyan")
    p = dir / "sim_mat_cosim.torch"
    if( p.exists()):
        cprint("sim_mat already exists","green")
        sim_mat = torch.load(p)
    else:
        cprint("sim_mat doesn't exist yet, calulating...", "yellow")
        sim_mat = calc_ip_cosine_sim(fts_stack=fts_stack_norm)
        torch.save(sim_mat, p)
        
    # cprint("euclidian distance", "cyan")
    # sim_mat = calc_eucl_dist_sim(fts_stack=fts_stack)
    # cprint("euclidian distance with normalized features", "cyan")
    # sim_mat = calc_eucl_dist_sim(fts_stack=fts_stack_norm)
    
    calc_avg_precision(sim_matrix=sim_mat, labels=labels, verbose=True)
    
    
def main():
    #Specify the model below! Possible options are:
    #"rotnet", "jigsaw", "moco", "simclr" and "swav"
    options = ["rotnet", "jigsaw", "moco", "simclr", "swav"]
    print(f"Choose a model to calculate simularity with the embeddinig gallary. Your options are: {options}")
    model_name = input("Your Choice:")
    while model_name not in options:
        print(f"Invalid option. Your options are: {options}")
        model_name = input("Your Choice:") 
    
    calc_sim_matrix(model_name)
    
if __name__ == "__main__":
    main()