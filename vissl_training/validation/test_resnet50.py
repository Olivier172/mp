import torchvision
import torch

# #lets get only the feature extraction
# resnet50_model = torchvision.models.resnet50(pretrained=True)
# print(resnet50_model._forward_impl)
# #handle = resnet50_model.register_forward_hook()

# #copy model without fc layer at the end,
# #output should be ftr vector with size 2048 
# model = torch.nn.Sequential(
#     resnet50_model.conv1,
#     resnet50_model.bn1,
#     resnet50_model.relu,
#     resnet50_model.maxpool,
#     resnet50_model.layer1,
#     resnet50_model.layer2,
#     resnet50_model.layer3,
#     resnet50_model.layer4,
#     resnet50_model.avgpool
# )

# #print(model)

#Absolute path to the checkpoints dir on this pc. The weights are stored here under dataset/model/weigths_file.torch
BASE_DIR_WEIGHTS = "/home/olivier/Documents/master/mp/checkpoints/"
#dictionary to summarize the paths to the the training config used and the path to the weigths
#train_config path is a relative path from the vissl folder
#weights path is an absolute path to where the final_checkpoint.torch is stored 
PATHS = {
    "rotnet":
    {
        "train_config": "validation/rotnet_full/train_config.yaml", #relative path from vissl/...
        "weights": BASE_DIR_WEIGHTS + "sku110k/rotnet_full/model_final_checkpoint_phase104.torch",
    },
    "jigsaw":
    {
        "train_config": "validation/jigsaw_full/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/jigsaw_full/model_final_checkpoint_phase104.torch"
    },
    "moco32":
    {
        "train_config": "validation/moco_full_32/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/moco_full_32/model_final_checkpoint_phase99.torch"
    },
    "moco64":
    {
        "train_config": "validation/moco_full_64/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/moco_full_64/model_final_checkpoint_phase99.torch"
    },
    "simclr":
    {
        "train_config": "validation/simclr_full/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/simclr_full/model_final_checkpoint_phase99.torch"
    },
    "swav":
    {
        "train_config": "validation/swav_full/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/swav_full/model_final_checkpoint_phase"
    },  
    #Models from different checkpoints:
    #ROTNET
    "rotnet_phase0":
    {
        "train_config": "validation/rotnet_full/train_config.yaml", 
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/rotnet_full/model_phase0.torch",
    },
    "rotnet_phase25":
    {
        "train_config": "validation/rotnet_full/train_config.yaml", 
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/rotnet_full/model_phase25.torch",
    },
    "rotnet_phase50":
    {
        "train_config": "validation/rotnet_full/train_config.yaml", 
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/rotnet_full/model_phase50.torch",
    },
    "rotnet_phase75":
    {
        "train_config": "validation/rotnet_full/train_config.yaml", 
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/rotnet_full/model_phase75.torch",
    },
    "rotnet_phase100":
    {
        "train_config": "validation/rotnet_full/train_config.yaml", 
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/rotnet_full/model_phase100.torch",
    },
    #JIGSAW
    "jigsaw_phase0":
    {
        "train_config": "validation/jigsaw_full/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/jigsaw_full/model_phase0.torch"
    },
    "jigsaw_phase25":
    {
        "train_config": "validation/jigsaw_full/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/jigsaw_full/model_phase25.torch"
    },
    "jigsaw_phase50":
    {
        "train_config": "validation/jigsaw_full/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/jigsaw_full/model_phase50.torch"
    },
    "jigsaw_phase75":
    {
        "train_config": "validation/jigsaw_full/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/jigsaw_full/model_phase75.torch"
    },
    "jigsaw_phase100":
    {
        "train_config": "validation/jigsaw_full/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/jigsaw_full/model_phase100.torch"
    },
    #MOCO32
    "moco32_phase0":
    {
        "train_config": "validation/moco_full_32/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/moco_full_32/model_phase0.torch"
    },
    "moco32_phase25":
    {
        "train_config": "validation/moco_full_32/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/moco_full_32/model_phase25.torch"
    },
    "moco32_phase50":
    {
        "train_config": "validation/moco_full_32/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/moco_full_32/model_phase50.torch"
    },
    "moco32_phase75":
    {
        "train_config": "validation/moco_full_32/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/moco_full_32/model_phase75.torch"
    },
    #SIMCLR
    "simclr_phase0":
    {
        "train_config": "validation/simclr_full/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/simclr_full/model_phase0.torch"
    },
    "simclr_phase25":
    {
        "train_config": "validation/simclr_full/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/simclr_full/model_phase25.torch"
    },
    "simclr_phase50":
    {
        "train_config": "validation/simclr_full/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/simclr_full/model_phase50.torch"
    },
    "simclr_phase75":
    {
        "train_config": "validation/simclr_full/train_config.yaml",
        "weights": "/home/olivier/Documents/master/mp/checkpoints/sku110k/simclr_full/model_phase75.torch"
    }
}

print(PATHS["rotnet"]["weights"])

#sanity check for avg embedding gallary
tensor = torch.Tensor([
    [1,2,3,4],
    [1,2,3,4],
    [1,2,3,4],
    [1,2,3,4],
    [1,2,3,4],
    [1,2,3,4],                  
    [1,2,3,4],        
])
mean = torch.mean(tensor, dim=0)
print(mean)