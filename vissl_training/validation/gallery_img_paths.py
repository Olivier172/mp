from pathlib import Path
from termcolor import cprint

def generate_gallery_paths(cornershop:Path, output_file:Path, relative_paths=False):
    #create an iterator over all jpg files in dataset_folder 
    img_paths = list(cornershop.glob("*/*.jpg")) #*/*.jpg look into all folders in this folder and search for files with extension .jpg
    gallery_img_paths = [str(p) for p in img_paths]
    
    if(relative_paths):
        gallery_img_paths = [p.replace(str(cornershop.parent.parent.parent) + "/", "") for p in gallery_img_paths]

    #write to file
    with open(output_file, "w") as f:
        f.writelines("\n".join(gallery_img_paths))
    

def main():
    cornershop = Path("/home/olivier/Documents/master/mp/CornerShop/CornerShop/crops")
    output_file = Path("data/gallery_paths_relative.txt")

    #Dont overwrite previously generated file:
    if output_file.is_file():
        cprint(f"Info: output_file already exists on path {output_file} so aborting", "yellow")
    
    #If file doesn't exist yet, generate
    generate_gallery_paths(cornershop=cornershop, output_file=output_file, relative_paths=True)
    
if __name__ == "__main__":
    main()
