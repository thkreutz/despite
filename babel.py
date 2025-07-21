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

# Define evaluation settings
freeze_options = [0, 1]  # Freeze model weights or not
projections = ["linear", "non-linear"]  # Projection head type

# Base evaluation command template
base_command = (
    "python evaluate_babel.py --model_type {model_type} --pretrained_path {pretrained_path} --dataset {dataset_type} "
    "--modality {modality} --freeze {freeze} --projection {projection} "
    "--n_frames 24 --batch_size 24 --epochs 35 --embed_dim 512 --wandb 1"
)

# Iterate over each model and its respective modalities
for (model_type, dataset_type), vals in run_dict.items():
    if dataset_type == "v1":
        continue
    modalities = vals["modalities"]
    pretrained_path = vals["pretrained_path"]
    
    # Evaluate all possible modality combinations for the given model (except for text)
    for modality in modalities:
        if modality == "text":
            continue

        # Evaluate all freeze-projection combinations
        for freeze, projection in itertools.product(freeze_options, projections):
            # Construct the command
            command = base_command.format(
                model_type=model_type,
                pretrained_path=pretrained_path,
                dataset_type=dataset_type,
                modality=modality,
                freeze=freeze,
                projection=projection
            )

            print(f"Running for Model Type: {model_type}, Modality: {modality}, Pretrained: {pretrained_path}, PreDataset Version : {dataset_type,}, Freeze: {freeze}, Projection: {projection}")
            os.system(command)  # Execute the command

print("All evaluations completed!")