from pathlib import Path
from termcolor import cprint
import json

def determine_query_origin(dataset_folder:Path):
    """
    Determines the origin of each embedding in the embedding gallery by iterating
    over the dataset. The resulting dict has a list of information about queries per class. 
    This info contains the gallery_idx and filename of the original image that was used to calculate the embedding.

    Args:
        dataset_folder (Path): Folder where the dataset is stored to construct embedding gallery.
        
    Returns:
        query_origins (dict): dict with information about queries for every class of the embedding gallery.
                              Per class a list is stored with the idx in the gallery and the filename of the original image.
                              e.g. query_origins = {
                                  "class1": [{"gallery_idx": 1, "filename": "p1"}, {"gallery_idx": 2, "filename": "p2"}]
                              }
    """
    query_origins = {}
    
    #create an iterator over all jpg files in dataset_folder 
    img_paths = dataset_folder.glob("*/*.jpg") #*/*.jpg look into all folders in this folder and search for files with extension .jpg
    for idx, p in enumerate(img_paths):
        p:Path
        #extract query information
        query_info = {
            "gallery_idx": idx,
            "filename": p.stem,
            "origin_img": p.stem.split("_")[0]
        }
        label = p.parent.stem
        
        #check if class label is already registered in query_origins dict
        if label not in query_origins.keys():
            #init with empty list of queries for this class
            query_origins[label] = []
        
        #add query info to its class
        query_origins[label].append(query_info)
        
    return query_origins

def train_test_split(queries:list, verbose=False):
    """
    Calculates a test and train set. 
    All queries are split up in 20% test, 80%train with the constraint that queries in the test set 
    come from different origin images then the ones in the train set.

    Args:
        queries (list): list of query dicts
        verbose (bool, optional): switch to turn on prints from this function.
        
    Returns:
        train_set (list): The training set.
        test_set (list): The test set for strict testing.
    """
    # Calculate the number of dictionaries for each set
    total_queries = len(queries)
    
    num_test = int(total_queries * 0.2) #20% of the total queries
    if(num_test < 1):
        num_test=1 # at least one
    num_train =  total_queries - num_test  # Remaining 80%
    if(verbose):
        print(f"\nnumtest {num_test}, numtrain {num_train}")

    # Initialize the sets
    train_set = []
    test_set = []

    #store all queries per origin img:
    queries_per_origin_img ={}
    for query in queries:
        origin_img = query["origin_img"]
        if origin_img not in queries_per_origin_img.keys():
            queries_per_origin_img[origin_img] = [query]
        else:
            queries_per_origin_img[origin_img].append(query)

    #try to find the best origin img to put its queries in the test set
    best_fit = {"diff": 200, "key": None}
    for k in queries_per_origin_img.keys():
        diff = abs(len(test_set) + len(queries_per_origin_img[k]) - num_test)
        if ( diff < best_fit["diff"]):
            best_fit["diff"] = diff
            best_fit["key"] = k

    #remove this set of items from one origin img and add it to the test_set
    best_set = queries_per_origin_img.pop(best_fit["key"])
    if(verbose):
        print(f"best diff {best_fit['diff']}")
    test_set.extend(best_set)
    
    #add all other queries to tain
    for k in queries_per_origin_img.keys():
        train_set.extend(queries_per_origin_img[k])

    if(verbose):
        # Print the resulting sets
        print("train_set:")
        for item in train_set:
            print(item)

        print("\ntest_set:")
        for item in test_set:
            print(item)
        
    return train_set, test_set

def check_eligibility_strict_testing(query_orgins: dict):
    """
    Function that searches for classes that are eligible for strict testing.
    Strict testing means that we have enough embeddings such that the origin images of the train crops are different
    from the origin images of the test crops.

    Args:
        query_orgins (dict): Dict with file and img origin information about all the queries of the embedding gallery per class.
        
    Returns:
        strict_train_test (dict): Dict with per class train test split for all the queries of eligible classes.
    """
    #store train and test gallery idxs per class to a dict:
    strict_train_test = {}
    
    for cls in query_orgins.keys():
        imgids = []
        queries = query_orgins[cls]
        for querie in queries:
            #save origin image, ignore cropid
            imgid, _ = querie["filename"].split("_")
            imgids.append(imgid)
            
        #calc the amount of distinct origin images for embeddings of this class
        amt_origin_imgs = len(set(imgids))
        if(amt_origin_imgs > 1):
            train_set, test_set = train_test_split(queries)
            #save train test split:
            strict_train_test[cls] = {
                "train": train_set, 
                "test": test_set 
            }
            #cprint(f"Info: class {cls} is eligible for strict testing with {amt_origin_imgs} distinct origin images", "yellow")
    return strict_train_test

def output_file_exists(output_file:Path):
    """
    Check if outputfile already exisits. 
    If so, no need to recalculate.

    Args:
        outputfile (Path): path to the output json file.
    """
    if(output_file.is_file()):
        cprint(f"Info: json file already exisits at {output_file} so aborting", "yellow")
        return True
    else:
        return False
        
def main():
    #input
    #The CornerShop dataset:
    cornershop = Path("/home/olivier/Documents/master/mp/CornerShop/CornerShop/crops")
    
    #output 
    #Json file that determines the strict train_test_split
    output_file = Path("data/strict_train_test.json")
    
    if(output_file_exists(output_file)):
        exit()
        
    qo = determine_query_origin(cornershop)
    strict_train_test = check_eligibility_strict_testing(qo)
    print(f"there are {len(strict_train_test.keys())} classes eligible for strict testing")   
    
    #save output
    with open(output_file, "w") as f:
        json.dump(strict_train_test, f)
     
    
if __name__ == "__main__":
    main()
    