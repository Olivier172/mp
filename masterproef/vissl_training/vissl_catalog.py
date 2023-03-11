from vissl.utils.io import save_file
from vissl.data.dataset_catalog import VisslDatasetCatalog
# run this file in the virtual env for vissl lib
# source .venv/bin/acivate

def main():
    #print the catalog
    print("catalog before appending to json")
    print(VisslDatasetCatalog.list(), end="\n\n")
    #you can find this catalog in vissl/configs/config/dataset_catalog.json

    json_data = {
        "Testfolder": {
            "train": ["<img_path_train>", "<lbl_path_train>"],
            "val": ["<img_path_val>", "<lbl_path_val>"]
        }
    }

    #save_file(json_data, "/home/olivier/Documents/mp/vissl/configs/config/dataset_catalog.json", append_to_json=True)
    print(VisslDatasetCatalog.list())
    #['imagenet1k_folder']
    #print(VisslDatasetCatalog.get("Testfolder"),end="\n\n")
    #{'train': ['<img_path>', '<lbl_path>'], 'val': ['<img_path>', '<lbl_path>']}

if __name__ == "__main__":
    main()