import os

def rename_files(parent_folder, old_part, new_part):
    for root, dirs, files in os.walk(parent_folder):
        for file_name in files:
            old_name = os.path.join(root, file_name)
            new_name = os.path.join(root, file_name.replace(old_part, new_part))
            
            if old_name != new_name:
                os.rename(old_name, new_name)
                print(f"Renamed: {old_name} -> {os.path.basename(new_name)}")

# Example usage
parent_folder = "data"  # Location of the galleries
old_part = 'gallary'  # part you want to replace
new_part = 'gallery'  # the new part

if __name__ == "__main__":
    #rename_files(parent_folder, old_part, new_part)
    pass
