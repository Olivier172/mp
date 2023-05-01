#old version before ten crop  1/05/2023  
def extract_features(path:Path, model, verbose=False) -> torch.Tensor:
    """calculates the inference of an image (from path) using the model given. Returns the feature vector.

    Args:
        path (Path): Path to the input image.
        model (vissl model): Vissl model to use for inference.
        verbose (bool, optional): When True, you get prints from this function. Defaults to False.

    Returns:
        torch.Tensor: a feature vector with dimensions torch.size([2048])
    """
    image = Image.open(path)
    # Convert images to RGB. This is important
    # as the model was trained on RGB images.
    image = image.convert("RGB")

    # Image transformation pipeline.
    # Must match the training transformation pipeline
    pipeline = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(255),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    x = pipeline(image)

    #unsqueeze adds a dim for batch size (with 1 element the entire input tensor of the image)
    features = model(x.unsqueeze(0))

    features_shape = features[0].shape
    if(verbose):
        print(f"Features extracted have the shape: { features_shape }")
    return features[0]