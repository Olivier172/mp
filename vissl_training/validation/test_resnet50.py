import torchvision
import torch

#lets get only the feature extraction
resnet50_model = torchvision.models.resnet50(pretrained=True)
print(resnet50_model._forward_impl)
#handle = resnet50_model.register_forward_hook()

#copy model without fc layer at the end,
#output should be ftr vector with size 2048 
model = torch.nn.Sequential(
    resnet50_model.conv1,
    resnet50_model.bn1,
    resnet50_model.relu,
    resnet50_model.maxpool,
    resnet50_model.layer1,
    resnet50_model.layer2,
    resnet50_model.layer3,
    resnet50_model.layer4,
    resnet50_model.avgpool
)

#print(model)