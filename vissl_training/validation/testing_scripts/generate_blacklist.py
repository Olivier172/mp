from pathlib import Path
from termcolor import cprint
from collections import Counter
import os
import json

def read_blacklist(file_path:Path, verbose=False) -> list:
    """
    Reads in the blacklist for class labels not to use when evaluating mAP scores of embedding_gallery_avg.
    Because these classes contain only one element in the CornerShop dataset and would be an "easy match".

    Args:
        file_path (Path): path to the blacklist file.
        verbose (bool, optional): print switch.

    Returns:
        blacklist (list): list of strings containing the blacklisted classes. 
    """
    if(not os.path.isfile):
        cprint(f"Warning: blacklist file for embedding_gallery_avg doesn't exist at {file_path}", "red")
        return []
    
    with open(file_path, "r") as f:
        blacklist = f.read().splitlines()
        if(verbose):
            cprint(f"Blacklist file succesfully read. There are  "+ str(len(blacklist)) + " classes blacklisted.", "green")
    return blacklist


def generate_blacklist(label_file:Path, blacklist_file:Path, verbose=False, generate_occurence_json=False):
    """
    Read in the label_file that represents all the labels that were used to create
    the embedding_gallery_avg. Classes that have only one image/label are put on the blacklist.
    This is done to prevent "easy matches" when evaluating mAP scores with embedding_gallery_avg.

    Args:
        label_file (Path): path to the label file of all the embeddings that were used to calculated an embedding_gallery_avg.
        blacklist_file (Path): file path to save the blacklist to.
        verbose (bool, optional): Controls prints.
    """
    if(not os.path.isfile(label_file)):
        cprint(f"label_file doesn't exist on path {label_file}", "red")
    
    #Read labels from file that contains all labels used in the embedding_gallery_avg  
    with open(label_file, "r") as f:
        labels = f.read().splitlines()
        
    #Lets count how many times each label occurs
    counter = Counter(labels)
    #list to store all elements that occur only once
    blacklist = list()
    #store the amount of occurences per class
    class_occurences = {} 
    for item, count in counter.items():
        class_occurences[item] = count
        if(verbose):
            print(f"{item} occurs {count} times.")
        if(count == 1):
            blacklist.append(item)
            if(verbose):
                cprint(f"Item {item} was added to the blacklist", "red")
                
    #Write results to output file
    with open(blacklist_file, "w") as f:
        cprint(f"Saving blacklist to file {blacklist_file}", "green")
        f.writelines("\n".join(blacklist))
        
    if(generate_occurence_json):
        occurence_json_file = blacklist_file.parent
        occurence_json_file /= "class_occurences.json"
        with open(occurence_json_file, "w") as f:
            cprint(f"Saving class occurences to file {occurence_json_file}", "green")
            json.dump(class_occurences, f)
        
def main():
    label_file = Path("data/rotnet/embedding_gallery_labels.txt")
    blacklist_file = Path("data/blacklist.txt")
    generate_blacklist(label_file=label_file, blacklist_file=blacklist_file, verbose=False, generate_occurence_json=True)
    
if __name__ == "__main__":
    main()