#vissl stuff
from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.models import build_model
from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_consolidated_weights

def load_model(modelName:str, verbose=False):
    #dictionary to summarize the paths to the the training config used and the path to the weigths
    #train_config path is a relative path from the vissl folder
    #weights path is an absolute path to where the final_checkpoint.torch is stored 
    PATHS = {
        "rotnet":
        {
            "train_config": "validation/rotnet_full/train_config.yaml", #relative path from vissl/...
            "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/rotnet_full/model_final_checkpoint_phase104.torch",
        },
        "jigsaw":
        {
            "train_config": "validation/jigsaw_full/train_config.yaml",
            "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/jigsaw_full/model_final_checkpoint_phase104.torch"
        },
        "moco":
        {
            "train_config": "validation/moco_full/train_config.yaml",
            "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/moco_full/model_final_checkpoint_phase199.torch"
        },
        "simclr":
        {
            "train_config": "",
            "weights": ""
        },
        "swav":
        {
            "train_config": "",
            "weights": ""
        }    
    }
    if(modelName not in PATHS.keys()):
        print(f"The model you tried to load ({modelName}) is not available")
        return 0

    #CHOOSE the model you want to validate here
    train_config = PATHS[modelName]["train_config"] #change the key of the PATHS dict to the desired model name
    weights_file = PATHS[modelName]["weights"]
    if(verbose):
        print('Train config at (relative path from vissl/...):\n' + train_config)
        print('SSL pretrained weights at:\n' + weights_file)
    
    # 1. Checkpoint config is located at vissl/configs/config/validation/*/train_config.yaml.
    # 2. weights are located at /home/olivier/Documents/master/mp/checkpoints/sku110k/*
    # The * in the above paths stand for rotnet_full, jigsaw_full or moco_full
    # All other options specified below override the train_config.yaml config.

    cfg = [
        'config=' + train_config,
        'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=' + weights_file, # Specify path for the model weights.
        'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
        'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
        'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
        'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
        'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["res5avg", ["Identity", []]]]' # Extract only the res5avg features.
    ]

    # Compose the hydra configuration.
    cfg = compose_hydra_configuration(cfg)
    # Convert to AttrDict. This method will also infer certain config options
    # and validate the config is valid.
    _, cfg = convert_to_attrdict(cfg)

    #build the model
    model = build_model(cfg.MODEL, cfg.OPTIMIZER)
    if(verbose):
        print(f"Model {modelName} was succusfully build")
    
    # Load the checkpoint weights.
    weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)

    # Initializei the model with the simclr model weights.
    init_model_from_consolidated_weights(
        config=cfg,
        model=model,
        state_dict=weights,
        state_dict_key_name="classy_state_dict",
        skip_layers=[],  # Use this if you do not want to load all layers
    )

    if(verbose):
        print(f"Weights for model {modelName} succesfully loaded")
        
    return model