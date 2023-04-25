import torch
import os
import torchvision.transforms as transforms
from load_vissl_model import load_model
from PIL import Image
from pathlib import Path
from termcolor import cprint #colored prints
from tqdm import tqdm #loading bars with heavy loops

def extract_features(path:Path, model, verbose=False) -> torch.Tensor:
    """calculates the inference of an image (from path) using the model given. Returns the feature vector.

    Args:
        path (Path): Path to the input image.
        model (vissl model): Vissl model to use for inference.
        verbose (bool, optional): When True, you get prints from this function. Defaults to False.

    Returns:
        torch.Tensor: _description_
    """
    image = Image.open(path)
    # Convert images to RGB. This is important
    # as the model was trained on RGB images.
    image = image.convert("RGB")

    # Image transformation pipeline.
    pipeline = transforms.Compose([
      transforms.CenterCrop(224),
      transforms.ToTensor(),
    ])
    
    x = pipeline(image)

    #unsqueeze adds a dim for batch size (with 1 element the entire input tensor of the image)
    features = model(x.unsqueeze(0))

    features_shape = features[0].shape
    if(verbose):
        print(f"Features extracted have the shape: { features_shape }")
    return features[0]


def make_embedding_gallary(dir:Path, model, verbose=False, exist_ok=False):
    """Generates an embedding gallary by calculating the features from all images of the CornerShop dataset with 
    the model provided.

    Args:
        dir (Path): Directory to save the gallary to.
        model (vissl model): Vissl model to use for inference.
        verbose (bool, optional): When True, you get prints from this function. Defaults to False.
        exist_ok (bool, optional): determines wether to overwrite an existing gallary or not
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
    fts_stack = torch.stack([extract_features(p,model).squeeze() for p in tqdm(img_paths)])
          
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
    model_name = "simclr"
    #Loading the model
    model = load_model(model_name,verbose=True)
    #Checking for GPU device
    device = torch.device("cpu")
    # if(torch.cuda.is_available()):
    #     device = torch.device("cuda")
    #     model.to(device) #move model to gpu
    print(f"using {device} device", end="\n\n")   
    
    #creating the embedding library
    make_embedding_gallary(Path("data/" + model_name),model, verbose=True, exist_ok=False)
    read_embedding_gallary(Path("data/" + model_name))
    
if __name__ == "__main__":
    main()