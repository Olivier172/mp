import torch
import pickle
import torchvision.transforms as transforms
from load_vissl_model import load_model
from PIL import Image
from pathlib import Path

    

def extract_features(path:Path, model, verbose=False) -> torch.Tensor:
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

def make_embedding_gallary(dir:Path, model, verbose=False):
    CornerShop = Path("/home/olivier/Documents/master/mp/CornerShop/CornerShop/crops")
    #create an iterator over all jpg files in cornershop map and put elements in a list
    img_paths = list(CornerShop.glob("*/*.jpg")) #**/*.jpg to look into all subdirs for jpgs and iterate over them
    #extract the corresponding labels (folder names)
    img_paths = img_paths[0:40] # limit amount for now
    labels = [p.parent.stem for p in img_paths ] #stem attr, conatins foldername 
    #path.stem=filename without extension
    #path.name=filename with extension
    fts_stack = torch.stack([extract_features(p,model).squeeze() for p in img_paths])
    if(verbose):
        print("make_embedding_gallary")
        print(f"feature stack has shape {fts_stack.shape}")
        print(labels, end="\n\n")
        
    #NORMALIZE features in feature stack:
    fts_stack_norm = fts_stack / fts_stack.norm(dim=1,keepdim=True) 
    
    #saving the calulated results
    torch.save(fts_stack, dir / "embedding_stack.torch")
    torch.save(fts_stack, dir / "embedding_stack_norm.torch")
    with open(dir / "embedding_labels.pkl", "wb") as f:
        pickle.dump(labels,f)
    
def read_embedding_gallary(dir:Path):
    print("read_embedding_gallary")
    fts_stack = torch.load(dir / "embedding_stack.torch")
    print(f"fts_stack has shape {fts_stack.shape}")
    fts_stack_norm = torch.load(dir / "embedding_stack_norm.torch")
    print(f"fts_stack_norm has shape {fts_stack_norm.shape}")
    labels = list()
    with open(dir / "embedding_labels.pkl", "rb") as f:
        labels = pickle.load(f)
        print(f"labels list has length "+ str(len(labels)))
        print(labels, end="\n\n")
    
        
    
def main():
    print("----embedding_lib.py----")
    #Loading the model
    model = load_model("rotnet",verbose=True)
    #Checking for GPU device
    device = torch.device("cpu")
    if(torch.cuda.is_available()):
        device = torch.device("cuda")
        model.to(device) #move model to gpu
    print(f"using {device} device", end="\n\n")
    #creating the embedding library
    make_embedding_gallary(Path("data"),model, verbose=True)
    read_embedding_gallary(Path("data"))
    
    
    
if __name__ == "__main__":
    main()