import vissl
import tensorboard
#import apex
import torch


json_data = {
    "dummy_data_folder": {
      "train": [
        "/mnt/e/AAA_MASTERPROEF/github/masterproef/vissl/simCLRtest/content/dummy_data/train", "/mnt/e/AAA_MASTERPROEF/github/masterproef/vissl/simCLRtest/content/dummy_data/train"
      ],
      "val": [
        "/mnt/e/AAA_MASTERPROEF/github/masterproef/vissl/simCLRtest/content/dummy_data/val", "/mnt/e/AAA_MASTERPROEF/github/masterproef/vissl/simCLRtest/content/dummy_data/val"
      ]
    }
}

# use VISSL's api to save or you can use your custom code.
from vissl.utils.io import save_file
save_file(json_data, "/home/olivier/vissl/configs/config/dataset_catalog.json", append_to_json=False)

#opvragen
from vissl.data.dataset_catalog import VisslDatasetCatalog
# list all the datasets that exist in catalog
print(VisslDatasetCatalog.list())

# get the metadata of dummy_data_folder dataset
print(VisslDatasetCatalog.get("dummy_data_folder"))
