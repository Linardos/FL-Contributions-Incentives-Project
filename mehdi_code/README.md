# Glioma Segmentation Pipeline

This repository contains a pipeline for the segmentation of Glioma by using nnUnet model.

## Overview

The pipeline processes multi-modal MRI scans (T1-weighted pre-contrast, T1-weighted post-contrast, T2-weighted, and T2-FLAIR) to produce segmentation masks of different tumor sub-regions.

The workflow functions as follows:
1.  **Reformat the BraTS data into Decathlon Format**: Renaming the data into Decathlon filenaming convention.
2.  **Extracting finger print from fine tunning dataset**:  reading and extracting the necesserary info from fine tunning dataset.
3.  **Adopt the nnUnet Plan configuration**: Read the plan config from the trained model and adopt it to the fine tunning dataset.
4.  **Prepreocessing finetune dataset**: With the adopted plan, finetunning dataset will be preprocessed.
5.  **Model training**: Loading the checkpoint from pretrained model and training the model (fine tunning).
6.  **Inference with fine tuned model**: Using the updated model in the inference phase on testing data.
7.  **Inference with pretrained model**: Using the pretrained model in the inference phasee on the testing data.

## Dependencies

Install nnUnet from:
*   [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
Set the environment variable as instructed in nnUnet:

e.g.

```bash
export nnUNet_raw="/media/fabian/nnUNet_raw"
export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
export nnUNet_results="/media/fabian/nnUNet_results"
```

## Data Preparation

The input data must follow the BraTS standard naming convention. Each subject's data should be in its own folder, containing four NIfTI files (`.nii.gz`) corresponding to the four MRI sequences:

*   `...-t1c.nii.gz`: T1-weighted post-contrast
*   `...-t1n.nii.gz`: T1-weighted pre-contrast (native)
*   `...-t2w.nii.gz`: T2-weighted
*   `...-t2f.nii.gz`: T2-FLAIR
*   `...seg.nii.gz`: segmentation mask

An example of the data format that this model works with is:
```
├── BraTS-GLI-00160-000
│   ├── BraTS-GLI-00160-000-seg.nii.gz
│   ├── BraTS-GLI-00160-000-t1c.nii.gz
│   ├── BraTS-GLI-00160-000-t1n.nii.gz
│   ├── BraTS-GLI-00160-000-t2f.nii.gz
│   └── BraTS-GLI-00160-000-t2w.nii.gz
├── BraTS-GLI-00185-000
│   ├── BraTS-GLI-00185-000-seg.nii.gz
│   ├── BraTS-GLI-00185-000-t1c.nii.gz
│   ├── BraTS-GLI-00185-000-t1n.nii.gz
│   ├── BraTS-GLI-00185-000-t2f.nii.gz
│   └── BraTS-GLI-00185-000-t2w.nii.gz
├── BraTS-GLI-00271-000
│   ├── BraTS-GLI-00271-000-seg.nii.gz
│   ├── BraTS-GLI-00271-000-t1c.nii.gz
│   ├── BraTS-GLI-00271-000-t1n.nii.gz
│   ├── BraTS-GLI-00271-000-t2f.nii.gz
│   └── BraTS-GLI-00271-000-t2w.nii.gz
├── BraTS-GLI-00288-000
│   ├── BraTS-GLI-00288-000-seg.nii.gz
│   ├── BraTS-GLI-00288-000-t1c.nii.gz
│   ├── BraTS-GLI-00288-000-t1n.nii.gz
│   ├── BraTS-GLI-00288-000-t2f.nii.gz
│   └── BraTS-GLI-00288-000-t2w.nii.gz

```



## Usage

An example of the usage of the pipeline is provided in `example.py` script.

1 - Download the sample data for training from [Here](https://drive.google.com/file/d/174HNeMKLG0weFFbib6LeM0RE__cGqD5N/view?usp=sharing)

Unzip the folder and move it to any arbitrary path.

2 - Download the pretrained model (770) from [Here](https://drive.google.com/file/d/1nUO9JV6PePhvFo17pHK7wwJFJ_NKqQ2G/view?usp=sharing)

Unzip the folder and move it to `nnUNet_results` directory.

3 - Download the config file of the pretrained model form [Here](https://drive.google.com/drive/folders/1PvSEv9QiLq4uBr1PO-WWgq8zku9EhLKH?usp=sharing)

Unzip the folder and move it `nnUNet_preprocessed` directory.

4 - Download the sample dataset for inference (already in Decathlon format) from [here](https://drive.google.com/file/d/1fZNt-rpBBS764Tmg0lW0TFqbHG1nSoIH/view?usp=sharing)

Unzip the folder and move it to any arbitrary path.

### Before running `example.py`:

1 - `RAW_DATA_PATH = "PATH_TO_SAMPLE_RAW_DATA"` This is the path you have the raw data in Decathlon format as mentioned.

2 - `PRETRAINED_DATASET_ID = 770` This is the ID of the pretrained model, it will be read from `nnUNet_results` (checkpoints) and `nnUNet_preprocessed` (config) directories.

3 - `FINETUNE_DATASET_ID = 666` This is an arbitrary model identified should be 3 digits, e.g. 666

4 - `FINETUNE_PLANS_ID = 'nnUNetPlans_finetune_from_brats'` This is the plan identifier for the new fine tunning model.

5 - `PRETRAINED_CHECKPOINT_PATH = '<nnUNet_results>/Dataset770_BraTSGLIPreCropRegion/nnUNetTrainer__nnUNetResEncUNetPlans__3d_fullres/fold_0/checkpoint_final.pth'` Checkpoint path of the pretrained model

6 - `FOLD = 0` Fold number to be trained => fine tuning dataset

7 - `N_EPOCHS = XX` Number of epochs to be trained

8 - `INIT_LR = 1e-3` Initial learning rate, recommded value 1e-3

9 - `INPUT_FOLDER_INFER = 'INPUT_FOLDER_INFER = 'PATH_TO_TESTING_DATA''` ABS Path to the folder where testing data is located in Decathlon format

10 - `OUTPUT_FOLDER_INFER_FINETUNE = 'PATH_TO_SAVE_RESULTS_FINETUNE'` ABS path to the folder where the results of testing data will be saved from the fine tunned model.

11 - `OUTPUT_FOLDER_INFER_PRETRAINED = 'PATH_TO_SAVE_RESULTS_PRETRAINED'` ABS path to the folder where the results of testing data will be saved from the fine tunned model.

The results of steps 10 and 11 can be compared later....