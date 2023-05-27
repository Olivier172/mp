import torch
import torchvision
import torchvision.transforms as transforms
from load_vissl_model import load_model
from PIL import Image
from pathlib import Path
from termcolor import cprint #colored prints
from tqdm import tqdm #loading bars with heavy loops
from collections import OrderedDict

def extract_features(img_path:Path, model, verbose=False, device="cpu") -> torch.Tensor:
    """
    calculates the inference of an image (from img_path) using the model given. Returns the feature vector.
    The image goes trough a resize, centercrop, totensor and normalize transformation. Subsequently inference is calculated
    and the feature vector is returned.

    Args:
        path (Path): Path to the input image.
        model (vissl model): Vissl model to use for inference.
        verbose (bool, optional): When True, you get prints from this function. Defaults to False.
        device (string or torch.device): When gpu is available, device=torch.device("cuda") is used for inference (not implemented yet)

    Returns:
        torch.Tensor: a feature vector with dimensions torch.size([2048])
    """
    image = Image.open(img_path)
    # Convert images to RGB. This is important
    # as the model was trained on RGB images.
    image = image.convert("RGB")

    # Image transformation pipeline.
    # Must match the training transformation pipeline
    pipeline = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(256),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    x = pipeline(image)
    #unsqueeze adds a dim for batch size (with 1 element, the entire input tensor of the image)
    with torch.no_grad():
        features = model(x.unsqueeze(0))[0] #the features itself are in the first element of a list
    features = features.squeeze() #squeeze out dimensions with 1 element, because they serve no purpose anymore
    if(verbose):
        print(f"Features extracted have the shape: { features.shape }")
    return features

def extract_features_tencrop(img_path:Path, model, verbose=False, device="cpu") -> torch.Tensor:
    """
    calculates the inference of an image (stored at img_path) using the model given. 
    Test time augmentation is used, 10 crops are taken and the average embedding is returned.
    Returns the feature vector torch.size([2048]).

    Args:
        img_path (Path): The image path for the image to feed to the neural net
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

def make_embedding_gallery(dir:Path, model, dataset_folder:Path, verbose=False, exist_ok=False, device="cpu", feature_hook_dict=None):
    """
    Generates an embedding gallery by calculating the features from all images of the dataset_folder (e.g. CornerShop) with 
    the model provided.

    Args:
        dir (Path): Directory to save the gallery to.
        model (vissl model): Vissl model to use for inference.
        dataset_folder (Path): path to the folder of the dataset. In this folder the images can be found in subfolders per class.
        verbose (bool, optional): When True, you get prints from this function. Defaults to False.
        exist_ok (bool, optional): determines wether to overwrite an existing gallery or not
        device (string or torch device): cpu of gpu for inference
        feature_hook_dict (OrderedDict or None): a dictionary that contains the activations of intermediary layers of the model. 
                                                 This is used to get the avg_pool activations from the img_net pretrained model
    """
    if(verbose):
        cprint("In function make_embedding_gallery()","green")
        
    #check if saving dir exists, otherwise make it
    if(dir.is_dir()):
        p = dir / "embedding_gallery.torch"
        if( p.exists()):
            if(exist_ok):
                cprint(f"gallery already exists, but exist_ok={exist_ok} so overwriting","yellow")
                #exist_ok=True, the gallery will be recalculated and overwritten...
            else:  
                cprint(f"gallery already exists, but exist_ok={exist_ok} so abort","yellow")
                return #dont overwrite if exist_ok=false
        else:
           cprint(f"gallery doesn't exist yet, creating one at {p}","magenta") 
    else:
        cprint(f"No gallery for this model exists yet, creating one at {dir}","magenta")
        dir.mkdir()
        
    
    #create an iterator over all jpg files in cornershop map and put elements in a list
    img_paths = list(dataset_folder.glob("*/*.jpg")) #**/*.jpg to look into all subdirs for jpgs and iterate over them
    #extract the corresponding labels (folder names)
    labels = [p.parent.stem for p in img_paths ] #stem attr of parent contains foldername == class name 
    
    print("Extracting features:")
    #First determine if we have to use hooks to get the activations of the embedding for a model that has a head network: (e.g. imgnet_pretrained and random)
    if(feature_hook_dict != None):
        if(verbose):
            cprint("Info: using hooks to get embedding activations from this model. (necessary for e.g. imgnet_pretrained and random)", "yellow")
        #Extract features from the img_net pretrained model with a hook on the avg pool layer
        avg_pool_features_list = []
        for p in tqdm(img_paths):
            #calculate forward pass
            _ = extract_features(p, model, device=device, verbose=False)
            #read activations in avgpool layer from hook
            avg_pool_features = feature_hook_dict["avgpool"].squeeze()
            #print(f"avgpool activations have shape {avg_pool_features.shape} and look like {avg_pool_features[0:2]}")
            avg_pool_features_list.append(avg_pool_features)
        embedding_gallery = torch.stack(avg_pool_features_list)
        
    else:
        #Extract features from a vissl model
        embedding_gallery = torch.stack([extract_features(p,model,device=device, verbose=False) for p in tqdm(img_paths)])
        
    #Normalize features in embedding_gallery
    embedding_gallery_norm = embedding_gallery / embedding_gallery.norm(dim=1,keepdim=True) 
            
    if(verbose):
        cprint("Saving embedding gallery", "green")
        
        print(f"fts_stack has shape {embedding_gallery.shape}")
        print(f"Minimum value {embedding_gallery.min()}\nMaximum value {embedding_gallery.max()}")
        print(f"4 example tensors from this stack:\n{embedding_gallery[0:5]}")
        
        print(f"fts_stack_norm has shape {embedding_gallery_norm.shape}")
        print(f"Minimum value {embedding_gallery_norm.min()}\nMaximum value {embedding_gallery_norm.max()}")
        print(f"4 example tensors from this stack:\n{embedding_gallery_norm[0:5]}")
        
        print(f"labels list has length "+ str(len(labels)))
        print(f"4 examples from the label list are: {labels[0:4]}", end="\n\n")
        
    #saving the calulated results 
    torch.save(embedding_gallery, dir / "embedding_gallery.torch")
    torch.save(embedding_gallery_norm, dir / "embedding_gallery_norm.torch")
    file_name = "embedding_gallery_labels.txt"
    with open(dir / file_name, "w") as f:
        f.writelines("\n".join(labels))

    
def read_embedding_gallery(dir:Path, verbose=False):
    """Reads and displays some characteritics about the embedding galary saved in the dir provided.

    Args:
        dir (Path): Path to the directory where the embedding gallery is saved.
                    There should be 3 files here: "embedding_gallery.torch", "embedding_gallery_norm.torch" 
                    and "embedding_gallery_labels.txt". 
        verbose (bool, optional): print switch.
                             
    Returns:
        embedding_gallery: The embedding gallery contains a stack of embeddings for which the label is known.
        embedding_gallery_norm: gallery with normalized embeddings.
        labels: Ground turth labels for every row (embedding) in the embedding gallery.
    """

    cprint("In function read_embedding_gallery()","green")
    
    embedding_gallery = torch.load(dir / "embedding_gallery.torch")
    if(verbose):
        print(f"fts_stack has shape {embedding_gallery.shape}")
        print(f"Minimum value {embedding_gallery.min()}\nMaximum value {embedding_gallery.max()}")
        print(f"4 example tensors from this stack:\n{embedding_gallery[0:5]}")

    embedding_gallery_norm = torch.load(dir / "embedding_gallery_norm.torch")
    if(verbose):
        print(f"fts_stack_norm has shape {embedding_gallery_norm.shape}")
        print(f"Minimum value {embedding_gallery_norm.min()}\nMaximum value {embedding_gallery_norm.max()}")
        print(f"4 example tensors from this stack:\n{embedding_gallery_norm[0:5]}")
    
    labels = list()
    with open(dir / "embedding_gallery_labels.txt", "r") as f:
        labels = f.read().splitlines()
        if(verbose):
            print(f"labels list has length "+ str(len(labels)))
            print(f"4 examples from the label list are: {labels[0:4]}", end="\n\n")
            
    return embedding_gallery, embedding_gallery_norm, labels
  
def add_feature_hooks(model: torch.nn.Module):
    """
    Return a dictionary that stores each layer's most recent output,
    using the layer's name as key.
    
    Args:
        model (nn.Module): The model to add the hooks to.
    """
    features = OrderedDict()

    def build_feature_hook(child_name):
        features.clear()
        def feature_hook(module, input, output):
            features[child_name] = output.detach().cpu()
        return feature_hook

    for name, module in model.named_children():
        module.register_forward_hook(build_feature_hook(name))

    return features
    
def main():
    
    #path to the cornershop dataset
    cornershop_folder = Path("/home/olivier/Documents/master/mp/CornerShop/CornerShop/crops")
    
    #Check if data folder already exists
    data_folder = Path("data")
    if(not(data_folder.is_dir())):
        cprint("Info: creating data folder in the current directory to store embedding gallaries", "yellow")
        #If not making a folder to store embedding gallaries
        data_folder.mkdir()
    
    #choose a model
    options = ["rotnet", "jigsaw", "moco32", "moco64", "simclr", "swav", "imgnet_pretrained", "random", "all",
               "rotnet_phase0", "rotnet_phase25",  "rotnet_phase50", "rotnet_phase75","rotnet_phase100",
               "jigsaw_phase0", "jigsaw_phase25",  "jigsaw_phase50", "jigsaw_phase75","jigsaw_phase100",
               "moco32_phase0", "moco32_phase25",  "moco32_phase50", "moco32_phase75",
               "moco64_phase0", "moco64_phase25",  "moco64_phase50", "moco64_phase75",
               "simclr_phase0", "simclr_phase25",  "simclr_phase50", "simclr_phase75",
               "swav_phase0", "swav_phase25",  "swav_phase50", "swav_phase75"]
    print(f"Choose a model to calculate embeddings. Your options are: {options}")
    model_name = input("Your Choice:")
    while model_name not in options:
        print(f"Invalid option. Your options are: {options}")
        model_name = input("Your Choice:")
        
    #Loading the model
    if(model_name == "all"):
        #Calculate embedding gallery for all models
        choice = input("At every checkpoint for all models? (y/N): ")
        if( choice != "y"):
            targets = ["rotnet", "jigsaw", "moco32", "moco64", "simclr", "swav"]
        else:
            targets = ["rotnet", "rotnet_phase0", "rotnet_phase25",  "rotnet_phase50", "rotnet_phase75","rotnet_phase100", 
                       "jigsaw", "jigsaw_phase0", "jigsaw_phase25",  "jigsaw_phase50", "jigsaw_phase75","jigsaw_phase100",
                       "moco32", "moco32_phase0", "moco32_phase25",  "moco32_phase50", "moco32_phase75",
                       "moco64", "moco64_phase0", "moco64_phase25",  "moco64_phase50", "moco64_phase75",
                       "simclr", "simclr_phase0", "simclr_phase25",  "simclr_phase50", "simclr_phase75" ,
                       "swav", "swav_phase0", "swav_phase25",  "swav_phase50", "swav_phase75"]
            
        for target in targets:
            cprint(f"Calculating embedding gallery for target {target}\n", "red")
            model = load_model(target, verbose=True)
            #Evaluation mode
            model = model.eval()
            #creating the embedding library
            make_embedding_gallery(
                Path("data/" + target), 
                model, 
                dataset_folder=cornershop_folder,
                verbose=True, 
                exist_ok=False, 
                device="cpu", 
                feature_hook_dict=None
            )
    else:
        #Calculate embedding gallery for one model
        if(model_name == "imgnet_pretrained"):
            #resnet50 with imgnet pretrained weights (supervised model)
            #https://pytorch.org/vision/0.9/models.html?highlight=resnet50#torchvision.models.resnet50
            model = torchvision.models.resnet50(pretrained=True)
            #place a hook on the last layer before classification (average pool)
            #output should be ftr vector with size 2048 (activations from the avgpool layer)
            feature_hook_dict = add_feature_hooks(model)
        elif(model_name == "random"):
            #resnet50 model without pre-trained weights:
            model = torchvision.models.resnet50(pretrained=False)
            #place hooks to get the embeddings later on at the avg pool layer
            feature_hook_dict = add_feature_hooks(model)
        else:
            feature_hook_dict=None #not needed when using vissl models
            #vissl model
            model = load_model(model_name, verbose=True)
            
        #Evaluation mode
        model = model.eval()
        
        #Checking for GPU device
        device = "cpu"
        # if(torch.cuda.is_available()):
        #     device = torch.device("cuda")
        #     model.to(device) #move model to gpu
        print(f"using {device} device", end="\n\n")   
        
        #creating the embedding library
        make_embedding_gallery(
            dir = Path("data/" + model_name),
            model = model, 
            dataset_folder = cornershop_folder,
            verbose = True, 
            exist_ok = False, 
            device = device, 
            feature_hook_dict = feature_hook_dict
        )
        read_embedding_gallery(Path("data/" + model_name), verbose=True)
    
if __name__ == "__main__":
    main()
    