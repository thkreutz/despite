import glob
import yaml
import os
import itertools

def get_run_dict(runs_path, checkpoint_num=145):
    files = glob.glob(os.path.join(runs_path, "*"))
    runs = [run for run in files if "run-" in run]
    run_dict = {}
    for run in runs:

        config_file = os.path.join(run, "files", "config.yaml")
        if os.path.exists(config_file):
            # Read YAML file
            with open(config_file, 'r') as stream:
                data_loaded = yaml.safe_load(stream)
            modalities = data_loaded["Modalities"]["value"]
            model_type = data_loaded["Model Type"]["value"]
            dataset_type = data_loaded["dataset"]["value"]

            run_dict[(model_type, dataset_type)] = {"modalities" : modalities, "pretrained_path" : os.path.join(run, "files", "model_%s.pth" % checkpoint_num)}
    return run_dict

run_dict = get_run_dict("results/ICCV/models/*/wandb/", checkpoint_num=145)

# Base evaluation command template
base_command = (
    "python evaluate_matching_multi_param.py --model_type {model_type} --pretrained_path {pretrained_path} --dataset {dataset_type} "
    "--src_modality {src_modality} --tgt_modality {tgt_modality} "
    "--n_frames 24 --embed_dim 512 --n_scenes 1000"
)

# Iterate over each model and its respective modalities
for (model_type, dataset_type), vals in run_dict.items():
    if dataset_type == "v2":
        continue
    modalities = vals["modalities"]
    modalities = [m for m in modalities if not m == "text"]
    pretrained_path = vals["pretrained_path"]
    print(modalities)
    # Evaluate all possible modality combinations for the given model (except for text)
    for src_modality in modalities:
        for tgt_modality in modalities:
            if src_modality == tgt_modality:
                continue
            
            command = base_command.format(
                model_type=model_type,
                pretrained_path=pretrained_path,
                dataset_type=dataset_type,
                src_modality=src_modality,
                tgt_modality=tgt_modality
            )
            
            print(f"Running for Model Type: {model_type}, Pretrained: {pretrained_path}, PreDataset Version : {dataset_type,}, Source Modality: {src_modality}, Target Modality: {tgt_modality}")
            os.system(command)  # Execute the command

print("All evaluations completed!")