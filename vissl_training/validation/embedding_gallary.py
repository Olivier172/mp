import torch
import os
import torchvision.transforms as transforms
from load_vissl_model import load_model
from PIL import Image
from pathlib import Path
from termcolor import cprint #colored prints
from tqdm import tqdm #loading bars with heavy loops

def extract_features(img_path:Path, model, verbose=False, device="cpu") -> torch.Tensor:
    """calculates the inference of an image (stored at img_path) using the model given. 
    Test time augmentation is used, 10 crops are taken and the average embedding is returned.
    Returns the feature vector torch.size([2048]).

    Args:
        img (Path): The image path for the image to feed to the neural net
        model (vissl model): Vissl model to use for inference.
        verbose (bool, optional): When True, you get prints from this function. Defaults to False.
        device (string or torch.device): When gpu is available, device=torch.device("cuda") is used for inference

    Returns:
        torch.Tensor: a feature vector with dimensions torch.size([2048])
    """
    image = Image.open(img_path)
    # Convert images to RGB. This is important
    # as the model was trained on RGB images.
    image = image.convert("RGB")

    # Image transformation pipeline.
    pipeline = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(256),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    #test time augmenting
    #take 10 crops
    width, height = image.size #Image size, in pixels. The size is given as a 2-tuple (width, height).
    ten_crops=transforms.functional.ten_crop(image, size=(height/2, width/2)) 
    #push them through the transform pipeline
    transformed_crops = [pipeline(crop) for crop in ten_crops]

    ##USE GPU if possible
    if( device != "cpu"):
        #move data to gpu
        transformed_crops = [crop.to(device) for crop in transformed_crops]
        #move model to gpu
        model.to(device)
    
    #unsqueeze adds a dim for batch size (with 1 element the entire input tensor of the image)
    features = [model(x.unsqueeze(0))[0] for x in transformed_crops] #send the 10 transfromed crops through the pipeline
    if(verbose):
        print(f"shape of the 10 embeddings from the 10 crops {torch.stack(features).shape}")
        
    #take the average embedding
    features_avg = torch.mean(torch.stack(features), dim=0)
    if(verbose):
        print(f"shape after averaging {features_avg.shape }")
    features_avg = features_avg.squeeze() #squeezing useless dimensions
    if(verbose):
        print(f"shape of features for returning: {features_avg.shape }")
    return features_avg


def make_embedding_gallary(dir:Path, model, verbose=False, exist_ok=False, device="cpu"):
    """Generates an embedding gallary by calculating the features from all images of the CornerShop dataset with 
    the model provided.

    Args:
        dir (Path): Directory to save the gallary to.
        model (vissl model): Vissl model to use for inference.
        verbose (bool, optional): When True, you get prints from this function. Defaults to False.
        exist_ok (bool, optional): determines wether to overwrite an existing gallary or not
        device (string or torch device): cpu of gpu for inference
    """
    if(verbose):
        cprint("In function make_embedding_gallary()","green")
        
    #check if saving dir exists, otherwise make it
    if(os.path.isdir(dir)):
        if(not(exist_ok)):
            cprint(f"Gallary already exists, but exist_ok={exist_ok} so abort","yellow")
            return #dont overwrite if exist_ok=false
        cprint(f"Gallary already exists, but exist_ok={exist_ok} so overwriting","yellow")
        #exist_ok=True, the gallary will be recalculated and overwritten...
    else:
        cprint(f"Gallary doesn't exist yet, creating one at {dir}","magenta")
        dir.mkdir()
        
    CornerShop = Path("/home/olivier/Documents/master/mp/CornerShop/CornerShop/crops")
    #create an iterator over all jpg files in cornershop map and put elements in a list
    img_paths = list(CornerShop.glob("*/*.jpg")) #**/*.jpg to look into all subdirs for jpgs and iterate over them
    #img_paths = img_paths[0:100] # limit amount for now
    #extract the corresponding labels (folder names)
    labels = [p.parent.stem for p in img_paths ] #stem attr, conatins foldername 
    #path.stem=filename without extension
    #path.name=filename with extension
    print("Extracting features:")
    fts_stack = torch.stack([extract_features(p,model,device=device) for p in tqdm(img_paths)])
          
    #NORMALIZE features in feature stack:
    fts_stack_norm = fts_stack / fts_stack.norm(dim=1,keepdim=True) 
    
    #saving the calulated results
    try:     
        if(verbose):
            print("Saving embedding gallary")
            
            print(f"fts_stack has shape {fts_stack.shape}")
            print(f"Minimum value {fts_stack.min()}\nMaximum value {fts_stack.max()}")
            print(f"4 example tensors from this stack:\n{fts_stack[0:5]}")
            
            print(f"fts_stack_norm has shape {fts_stack_norm.shape}")
            print(f"Minimum value {fts_stack_norm.min()}\nMaximum value {fts_stack_norm.max()}")
            print(f"4 example tensors from this stack:\n{fts_stack_norm[0:5]}")
            
            print(f"labels list has length "+ str(len(labels)))
            print(f"4 examples from the label list are: {labels[0:4]}", end="\n\n")
        torch.save(fts_stack, dir / "embedding_gallary.torch")
        torch.save(fts_stack_norm, dir / "embedding_gallary_norm.torch")
        with open(dir / "embedding_gallary_labels.txt", "w") as f:
            f.writelines("\n".join(labels))
    except Exception:
        print(f"Something went wrong while saving embedding gallary, check the paths (dir={dir})")
    
def read_embedding_gallary(dir:Path):
    """Reads and displays some characteritics about the embedding galary saved in the dir provided.

    Args:
        dir (Path): Path to the embedding gallary.
    """
    try:
        cprint("In function read_embedding_gallary()","green")
        
        fts_stack = torch.load(dir / "embedding_gallary.torch")
        print(f"fts_stack has shape {fts_stack.shape}")
        print(f"Minimum value {fts_stack.min()}\nMaximum value {fts_stack.max()}")
        print(f"4 example tensors from this stack:\n{fts_stack[0:5]}")
        
        fts_stack_norm = torch.load(dir / "embedding_gallary_norm.torch")
        print(f"fts_stack_norm has shape {fts_stack_norm.shape}")
        print(f"Minimum value {fts_stack_norm.min()}\nMaximum value {fts_stack_norm.max()}")
        print(f"4 example tensors from this stack:\n{fts_stack_norm[0:5]}")
        
        labels = list()
        with open(dir / "embedding_gallary_labels.txt", "r") as f:
            labels = f.read().splitlines()
            print(f"labels list has length "+ str(len(labels)))
            print(f"4 examples from the label list are: {labels[0:4]}", end="\n\n")
    except Exception:
        print(f"Unable to read embedding gallary, check paths (dir={dir})")
    
def main():
    #Specify the model below! Possible options are:
    #"rotnet", "jigsaw", "moco", "simclr" and "swav"
    options = ["rotnet", "jigsaw", "moco", "simclr", "swav"]
    print(f"Choose a model to calculate embeddings. Your options are: {options}")
    model_name = input("Your Choice:")
    while model_name not in options:
        print(f"Invalid option. Your options are: {options}")
        model_name = input("Your Choice:")
    #Loading the model
    model = load_model(model_name,verbose=True)
    #Checking for GPU device
    device = "cpu"
    # if(torch.cuda.is_available()):
    #     device = torch.device("cuda")
    #     model.to(device) #move model to gpu
    print(f"using {device} device", end="\n\n")   
    
    #creating the embedding library
    make_embedding_gallary(Path("data/" + model_name),model, verbose=True, exist_ok=True, device=device)
    read_embedding_gallary(Path("data/" + model_name))
    
if __name__ == "__main__":
    main()
    