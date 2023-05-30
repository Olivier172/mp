from pathlib import Path

import re
import json
import matplotlib.pyplot as plt

def test_regex():
    regex_pattern = r"mAP=([\d.]+)% for [\w\s]+\((\w+)\) as a similarity metric."
    line = str("mAP=8.863552816400906% for inner product (ip) as a similarity metric.")
    match = re.match(regex_pattern, line)
    if match:
        print("match")
        score = float(match.group(1))
        metric = match.group(2)
        print(f"score {score} metric {metric}")

def parse_total_mAP_log(log_file:Path):
    """
    Parse a total mAP scores log to a json containen models > galleries > metrics: scores
    Args:
        log_file (Path): path to log you want to parse into a json dict.

    Returns:
        model_logs: dict with the model_names as keys. 
        Containing a dict with both galleries and in there a dict with all the metrics.
    """
    with open(log_file, 'r') as file:
        lines = file.readlines()

    model_logs = {}
    model_name = None
    gallery_data = {}
    regex_pattern = r"mAP=([\d.]+)% for [\w\s]+\((\w+)\) as a similarity metric."

    for line in lines:
        line = line.strip()
        
        if line.startswith("Results for"):
            model_name = line.split()[2]
            if(model_name not in model_logs.keys()):
                #register model
                model_logs[model_name] = {}
        if line.startswith("origin_file:"):
            #new gallery, clear data
            gallery_data = {}
            gallery_name = line.split(": ")[1]
            if "embedding_gallery_avg" in gallery_name:
                gallery_key = "embedding_gallery_avg"
            elif "embedding_gallery" in gallery_name:
                gallery_key = "embedding_gallery"
        elif line.startswith("mAP"):
            match = re.match(regex_pattern, line)
            if match:
                score = float(match.group(1))
                metric = match.group(2)
                gallery_data[metric] = score
                if(len(gallery_data) == 4):
                    # all metrics collected, save gallery_data
                    model_logs[model_name][gallery_key] = gallery_data
        else:
            continue
        

    return model_logs

def save_to_json(json_data, json_file):
    #print(json_data)
    with open(json_file, 'w') as file:
        json.dump(json_data, file, indent=4)
        
def plot_mAP_metric_progression(json_file, models, gallery, metric, output_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    markers = ['o', 's', '^', 'd', 'v', 'p', '*']  # Marker styles for different models

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        x_values = []
        y_values = []
        

        if(model in ["rotnet", "jigsaw"] ):
            checkpoint_phases = [0, 25, 50, 75, 100]
            final_checkpoint_nr = 104
        else:
            checkpoint_phases = [0, 25, 50, 75]
            final_checkpoint_nr = 99

        for nr in checkpoint_phases:
            checkpoint_name = model + f"_phase{nr}"
            x_values.append(nr)
            y_values.append(data[checkpoint_name][gallery][metric])
        #final checkpoint
        checkpoint_name = model
        x_values.append(final_checkpoint_nr)
        y_values.append(data[checkpoint_name][gallery][metric])

        plt.plot(x_values, y_values, marker=markers[i % len(markers)], label=model)

    plt.xlabel('epochs')
    plt.ylabel('mAP Score')
    plt.title(f'Progression of mAP score (metric = {metric}, gallery = {gallery})')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
            
         
def parse_total_perf_eval_log(log_file:Path):
    """
    Parse a total perf eval log to a json containing models > train_test_split > acc: value, mAP: value
    Args:
        log_file (Path): path to log you want to parse into a json dict.

    Returns:
        model_logs: dict with the model_names as keys. 
        Containing a dict with both train_test_splits and in there a dict with acc and mAP scores.
    """
    with open(log_file, 'r') as file:
        lines = file.readlines()

    model_logs = {}
    model_name = None
    scores = {}
    regex_pattern_acc = r"acc=([\d.]+)%"
    regex_pattern_mAP = r"mAP=([\d.]+)%"
    

    for line in lines:
        line = line.strip()
        
        if line.startswith("Results for"):
            model_name = line.split()[2]
            if(model_name not in model_logs.keys()):
                #register model
                model_logs[model_name] = {}
        if line.startswith("origin_file:"):
            #new train_test_split, clear data
            scores = {}
            gallery_name = line.split(": ")[1]
            if "perf_eval_strict" in gallery_name:
                train_test_split = "train_test_split_strict"
            elif "perf_eval" in gallery_name:
                train_test_split = "train_test_split_std"
        elif line.startswith("acc"):
            match = re.match(regex_pattern_acc, line)
            if match:
                acc = float(match.group(1))
                scores["acc"]=acc
                    
        elif line.startswith("mAP"):
            match = re.match(regex_pattern_mAP, line)
            if match:
                mAP = float(match.group(1))
                scores["mAP"] = mAP
                if(len(scores) == 2):
                    model_logs[model_name][train_test_split] = scores
        else:
            continue
        

    return model_logs

def plot_perf_eval_progression(json_file, models, train_test_split, metric, output_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    markers = ['o', 's', '^', 'd', 'v', 'p', '*']  # Marker styles for different models

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        x_values = []
        y_values = []
        

        if(model in ["rotnet", "jigsaw"] ):
            checkpoint_phases = [0, 25, 50, 75, 100]
            final_checkpoint_nr = 104
        else:
            checkpoint_phases = [0, 25, 50, 75]
            final_checkpoint_nr = 99

        for nr in checkpoint_phases:
            checkpoint_name = model + f"_phase{nr}"
            x_values.append(nr)
            y_values.append(data[checkpoint_name][train_test_split][metric])
        #final checkpoint
        checkpoint_name = model
        x_values.append(final_checkpoint_nr)
        y_values.append(data[checkpoint_name][train_test_split][metric])

        plt.plot(x_values, y_values, marker=markers[i % len(markers)], label=model)

    if(metric == "acc"):
        metric="accuracy" #for printing in axis and title
    plt.xlabel('epochs')
    plt.ylabel(f'{metric} Score')
    plt.title(f'Progression of {metric} score (split = {train_test_split})')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
  
def process_mAP_logs():
    # PARSING mAP logs to json
    f_name = "total_gallery_mAP_scores_log"
    log_mAP_txt = Path(f"thesis_logs/{f_name}.txt")
    log_mAP_json = Path(f"thesis_logs/{f_name}.json")
    
    json_data = parse_total_mAP_log(log_mAP_txt)
    save_to_json(json_data, log_mAP_json)
    
    #PLOTTING
    f_name = "total_gallery_mAP_scores_log"
    log_mAP_json = Path(f"thesis_logs/{f_name}.json")
    models = ["rotnet", "jigsaw", "moco32", "moco64", "simclr", "swav"]
    galleries = ["embedding_gallery", "embedding_gallery_avg"]
    metrics = ["ip", "cosim", "eucl_dist", "eucl_dist_norm"]
    
    for gallery in galleries:
        output_folder = Path(f"thesis_logs/{gallery}_mAPs")
        if(not output_folder.is_dir()):
            output_folder.mkdir()
        for metric in metrics:
            output_file = output_folder / metric
            plot_mAP_metric_progression(log_mAP_json, models, gallery, metric, output_file)  
            
def process_perf_eval_logs():
    
    # PARSING perf_eval logs to json
    #mlp
    f_name = "total_perf_eval_log_mlp"
    log_perf_eval_txt = Path(f"thesis_logs/{f_name}.txt")
    log_perf_eval_json = Path(f"thesis_logs/{f_name}.json")
    
    json_data = parse_total_perf_eval_log(log_perf_eval_txt)
    save_to_json(json_data, log_perf_eval_json)
    
    #svm
    f_name = "total_perf_eval_log_svm"
    log_perf_eval_txt = Path(f"thesis_logs/{f_name}.txt")
    log_perf_eval_json = Path(f"thesis_logs/{f_name}.json")

    json_data = parse_total_perf_eval_log(log_perf_eval_txt)
    save_to_json(json_data, log_perf_eval_json)
    
    #PLOTTING
    #mlp & svm
    classifiers = ["mlp", "svm"]
    for clf in classifiers:
        f_name = f"total_perf_eval_log_{clf}"
        log_perf_eval_json = Path(f"thesis_logs/{f_name}.json")
        models = ["rotnet", "jigsaw", "moco32", "moco64", "simclr", "swav"]
        train_test_splits = ["train_test_split_std", "train_test_split_strict"]
        metrics = ["acc", "mAP"]
        
        output_folder = Path(f"thesis_logs/perf_eval_{clf}_graphs")
        if(not output_folder.is_dir()):
                output_folder.mkdir()

        for train_test_split in train_test_splits:    
            for metric in metrics:
                output_file = output_folder / (train_test_split + "_" + metric)
                plot_perf_eval_progression(log_perf_eval_json, models, train_test_split, metric, output_file) 
    
    
def main():
    print("log_processing: log_to_json and json_to_plot :)")
    #process_mAP_logs()
    #process_perf_eval_logs()
    
if __name__ == "__main__":
    main()