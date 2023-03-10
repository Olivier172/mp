# # Install pytorch version 1.8
# !pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install Apex by checking system settings: cuda version, pytorch version, and python version
import sys
import torch
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{torch.__version__[0:5:2]}"
])
print(version_str)

# install apex (pre-compiled with optimizer C++ extensions and CUDA kernels)
# !pip install apex -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/{version_str}/download.html
# GEBRUIK DEZE VOOR APEX TE INSTALLEREN
# pip install apex -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py38_cu101_pyt180/download.html

#commando's:
# 1) pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# 2) pip install apex -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py38_cu101_pyt180/download.html
# 3) vissl
# # # clone vissl repository and checkout latest version.
# !git clone --recursive https://github.com/facebookresearch/vissl.git

# %cd vissl/

# !git checkout v0.1.6
# !git checkout -b v0.1.6

# # install vissl dependencies
# !pip install --progress-bar off -r requirements.txt
# !pip install opencv-python

# # update classy vision install to commit compatible with v0.1.6
# !pip uninstall -y classy_vision
# !pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/4785d5ee19d3bcedd5b28c1eb51ea1f59188b54d

# # Update fairscale to commit compatible with v0.1.6
# !pip uninstall -y fairscale
# !pip install fairscale@https://github.com/facebookresearch/fairscale/tarball/df7db85cef7f9c30a5b821007754b96eb1f977b6

# # install vissl dev mode (e stands for editable)
# !pip install -e .[dev]

# Deze dependencies zouden beschikbaar moeten zijn :
# import vissl
# import tensorboard
# import apex
# import torch

#trouble shooting
# als pip install -e .[dev] faalt omwille van Python.h
# sudo apt-get install python3.8-dev

#oude python installeren
# !sudo add-apt-repository ppa:deadsnakes/ppa
# !sudo apt update
# !sudo apt install python3.8
# !/usr/local/bin/python3.8 -m pip install virtualenv
