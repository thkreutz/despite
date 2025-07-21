# DeSPITE: Exploring Contrastive Deep Skeleton-Pointcloud-IMU-Text Embeddings for Advanced Point Cloud Human Activity Understanding

Public Repository for our ICCV 2025 accepted paper: DeSPITE: Exploring Contrastive Deep Skeleton-Pointcloud-IMU-Text Embeddings for Advanced Point Cloud Human Activity Understanding 

[Project Page](https://thkreutz.github.io/projects/despite.html) - [Arxiv](https://arxiv.org/abs/2506.13897)

## Authors
Thomas Kreutz, Max Mühlhäuser, and Alejandro Sanchez Guinea

## Abstract
Despite LiDAR (Light Detection and Ranging) being an effective privacy-preserving alternative to RGB cameras to perceive human activities, it remains largely underexplored in the context of multi-modal contrastive pre-training for human activity understanding (e.g., human activity recognition (HAR), retrieval, or person re-identification (RE-ID)). To close this gap, our work explores learning the correspondence between LiDAR point clouds, human skeleton poses, IMU data, and text in a joint embedding space. More specifically, we present DeSPITE, a Deep Skeleton-Pointcloud-IMU-Text Embedding model, which effectively learns a joint embedding space across these four modalities. At the heart of our empirical exploration, we have combined the existing LIPD and Babel datasets, which enabled us to synchronize data of all four modalities, allowing us to explore the learning of a new joint embedding space. Our experiments demonstrate novel human activity understanding tasks for point cloud sequences enabled through DeSPITE, including Skeleton<->Pointcloud<->IMU matching, retrieval, and temporal moment retrieval. Furthermore, we show that DeSPITE is an effective pre-training strategy for point cloud HAR through experiments in MSR-Action3D and HMPEAR. 

## Code & Data

### Python Requirements

- python 3.8.10
- pytorch
- cuda
- https://github.com/hehefan/PST-Transformer.git
- pip install git+https://github.com/openai/CLIP.git
- pip install einops
- pip install pandas
- pip install wandb
- pip install transformers==4.39.3
- PST-Transformer setup here: https://github.com/hehefan/PST-Transformer/tree/main, make sure to have modules after installation in src/modules of this repo
- For SMPL, follow https://github.com/4DVLab/LIP or https://github.com/GuyTevet/MotionCLIP/ to setup

### Datasets
- LIPD: download dataset here https://drive.google.com/file/d/1SVO77FWFUOtC-Et2sdlgAdiNQWzgzxt0/view?pli=1
- Babel: https://babel.is.tue.mpg.de/
- AMASS: https://amass.is.tue.mpg.de/
- MSR-Action3D: refer to the Meteornet Repo for download and setup https://github.com/xingyul/meteornet/tree/master/action_cls
- HMPEAR: dataset can be found here http://www.lidarhumanmotion.net/hmpear/

### Dataset Preprocessing
- First, set paths accordingly for where to find and store the data in src/dataset/lipd_preprocessing.py, src/dataset/lipdbabelamass_preprocessing.py, src/dataset/hmpear_preprocessing.py.
In the repo, datasets are assumed to be stored in a top-level /data directory like this /data/<LIPD, AMASS, BABEL, MSRAction3D, HMPEAR>. Just check the scripts, its straightforward you can change however you want.
- Step 1: Run python src/dataset/lipd_preprocessing.py to preprocess the LIPD dataset to get LIPD_SEQUENCES_256p.pkl (ps: I uploaded this data here [TUDataLib](https://tudatalib.ulb.tu-darmstadt.de/items/17c47531-5e6d-4c86-a685-740d8f94f398))
- Step 2: Run python src/dataset/lipdbabelamass_preprocessing.py to preprocess the LIPD dataset to get lipd_babel_annotations.pkl and lipd_babel_val.pkl (not possible to share due to license of AMASS and Babel so need to download them first and run this script)
- (Optional HAR HMPEAR) Step 3: Run python src/dataset/hmpear_preprocessing.py to get hmpear_train_1024_24f.pt and hmpear_test_1024_24f.pt required for HAR finetuning (not possible to share due to license of HMPEAR so need to download them first and run this script)
- (Optional HAR MSRAction3D) Step 4: Run python preprocess_file.py --input_dir /path/to/Depth --output_dir processed_data --num_cpu 11 (https://github.com/xingyul/meteornet/tree/master/action_cls to get the MSRAction3D/data folder), required for HAR finetuning

### Train joint embedding space
Only Step 1 and Step 2 of dataset preprocessing is needed to train the models. Basic training with default parameters as follows.

- To train DeSPITE on LIPD-Babel-v1, use skeleton pc imu text as modalities:
```
python train_SPITE.py --modalities skeleton pc imu text --embed_dim 512 --dataset v1 --with_generator 0 --batch_size 1024
```

- To train DeSPIE on LIPD-Babel-v1, use skeleton pc imu as modalities:
```
python train_SPITE.py --modalities skeleton pc imu --embed_dim 512 --dataset v1 --with_generator 0 --batch_size 1024
```

To trian on LIPD-Babel-v2, change --dataset v1 to --dataset v2

### Evaluations
For a specific evaluation, run the respective evaluate_*.py scripts and specify model params and other parameters you want. Model weights see next section.

Examples: 
Matching Evaluation
```
python evaluate_matching_multi_param.py --model_type {model_type} --pretrained_path {pretrained_path} --dataset {dataset_type} --src_modality {src_modality} --tgt_modality {tgt_modality} --n_frames 24 --embed_dim 512 --n_scenes 1000
```

Temporal Moment Retrieval Evaluation
```
python evaluate_temporal_localization_multi_param.py --model_type {model_type} --pretrained_path {pretrained_path} --dataset {dataset_type} --src_modality {src_modality} --tgt_modality {tgt_modality} --n_frames 24 --embed_dim 512
```

HAR finetuning on MSRAction3D
```
python evaluate_msr.py --model_type {model_type} --pretrained_path {pretrained_path} --dataset {dataset_type} --modality {modality} --freeze {freeze} --projection {projection} --n_frames 24 --batch_size 24 --epochs 35 --embed_dim 512 --wandb 0
```

HAR finetuning on HMPEAR 
```
python evaluate_hmpear.py --model_type {model_type} --pretrained_path {pretrained_path} --dataset {dataset_type} --modality {modality} --freeze {freeze} --projection {projection} --n_frames 24 --batch_size 24 --epochs 35 --embed_dim 512 --wandb 0
```

HAR finetuning on LIPD-Babel-v2 
```
python evaluate_babel.py --model_type {model_type} --pretrained_path {pretrained_path} --dataset {dataset_type} --modality {modality} --freeze {freeze} --projection {projection} --n_frames 24 --batch_size 24 --epochs 35 --embed_dim 512 --wandb 0
```


If you want to reproduce a full run of everything I reported in the paper, run the *.py scripts for the respective evaluation on all my models I uploaded to [TUDataLib](https://tudatalib.ulb.tu-darmstadt.de/items/17c47531-5e6d-4c86-a685-740d8f94f398). (see next section, ICCV_experiments.zip)

### Model weights
All model weights can be found here on [TUDataLib](https://tudatalib.ulb.tu-darmstadt.de/items/17c47531-5e6d-4c86-a685-740d8f94f398): 
- ICCV_experiments.zip includes all model weights at different epochs and all experimental results during training tracked with wandb or produced by the evaluation scripts, which also includes all HAR models.
- OpenAccess_models.zip includes only the final models of the joint embedding space, named by the respective version (v1 or v2) and the included modalities (e.g., skeleton, pointcloud, imu, text is SPITE) => Use these models if you want to do something.

### Demo
See the Demo.ipynb jupyter notebook to see how to use the pre-trained models for, e.g., RE-ID or temporal moment retrieval and how to produce some videos with it. 
Only needs the preprocessed LIPD data (LIPD_SEQUENCES_256p.pkl, link to download [TUDataLib](https://tudatalib.ulb.tu-darmstadt.de/items/17c47531-5e6d-4c86-a685-740d8f94f398)) and some model weights from e.g., DeSPITE or DeSPIE (link to download [TUDataLib](https://tudatalib.ulb.tu-darmstadt.de/items/17c47531-5e6d-4c86-a685-740d8f94f398)), so you can skip everything else around amass, babel, hmpear, msraction3d, smpl if you are just interested in this. 


