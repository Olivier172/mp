from pathlib import Path
import json

def main():
    print("halllo")
    p = Path("data")
    logfiles = p.glob("*/cross_val_strict_svm.txt")
    for logfile in logfiles:
        if(logfile.parent.stem == "rotnet"):
            print("skip")
            continue
        json_file_name = "cross_val_strict_svm_best_params.json"
        json_file = logfile.parent / json_file_name
        #deleting a wrong file
        # json_file_del = "cross_val_svm_strict_best_params.json"
        # json_file_del = logfile.parent / json_file_del
        # json_file_del.unlink()
        print(logfile)
        print(json_file)
        
        with open(logfile, "r") as f:
            lines = f.readlines()
        best_params = json.loads((lines[3][12:]).replace("\'","\""))
        print(best_params)
        print(type(best_params))
        
        
        with open(json_file, "w") as f:
            json.dump(best_params, f)
        
if __name__ == "__main__":
    #main()
    pass