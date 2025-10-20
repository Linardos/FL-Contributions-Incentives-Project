import os
import torch
import shutil
from nnunet_api import NnUnetApi
from tools.data_reformat import data_prepare
from tools.json_pickle_stuff import copy_plans_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results, nnUNet_raw

# =================================================================================
# 1. Configuration
# =================================================================================
# --- Define the IDs and paths for your datasets and models ---

# ABS path to the raw data directory: each subject has its own folder with BraTS namind convention
RAW_DATA_PATH = "PATH_TO_SAMPLE_RAW_DATA"

# The Dataset ID of the model you want to use for pre-training.
# This model's architecture and plans will be transferred to your new dataset.
PRETRAINED_DATASET_ID = 770

# The Dataset ID of your new dataset that you want to fine-tune on.
# Make sure you have already converted this dataset to the nnU-Net format.
FINETUNE_DATASET_ID = 666

# An identifier for the new plans that will be created for your fine-tuning dataset.
# It's good practice to give it a descriptive name.
FINETUNE_PLANS_ID = 'nnUNetPlans_finetune_from_brats'

# The full path to the pre-trained model's checkpoint file.
# This file contains the weights that will be used to initialize your new model.
PRETRAINED_CHECKPOINT_PATH = '<nnUNet_results>/Dataset770_BraTSGLIPreCropRegion/nnUNetTrainer__nnUNetResEncUNetPlans__3d_fullres/fold_0/checkpoint_final.pth'

# The GPU device to use for training.
# DEVICE = torch.device('cuda')

# Fold number you want to train (on the fine tunning dataset)
FOLD = 0

# NUMBER of epochs to train the model
N_EPOCHS = 3

# Initial Learning Rate for model training
INIT_LR = 1e-3

# ABS Path to the folder where testing data is located
INPUT_FOLDER_INFER = 'PATH_TO_TESTING_DATA'

# ABS Path to the folder where the results of inference will be saved
OUTPUT_FOLDER_INFER_FINETUNE = 'PATH_TO_SAVE_RESULTS_FINETUNE'
OUTPUT_FOLDER_INFER_PRETRAINED = 'PATH_TO_SAVE_RESULTS_PRETRAINED'


if __name__ == '__main__':
    # =================================================================================
    # 2. Initialize the API
    # =================================================================================
    api = NnUnetApi()

    # # =================================================================================
    # # Fine-Tuning Workflow
    # # =================================================================================
    # ## --- Step 00: Preparing the raw dataset into Decathlon format
    print("\nStep 0: Running Data Preparation script...")
    DST_DATA_NAME = "Dataset"+str(FINETUNE_DATASET_ID)+"_finetune"
    data_prepare(RAW_DATA_PATH, os.path.join(nnUNet_raw, DST_DATA_NAME))
    n_case=len(os.listdir(os.path.join(nnUNet_raw, DST_DATA_NAME,"labelsTr")))
    copy_plans_json("./dataset.json", os.path.join(nnUNet_raw, DST_DATA_NAME), n_case)


    ## --- Step 1: Extract the Fingerprint for Your New Dataset ---
    ## This step analyzes the properties of your new dataset (image sizes, spacings, etc.)
    ## and creates a "fingerprint" file. This is a prerequisite for any planning.
    ## This wraps: nnUNetv2_extract_fingerprint CLI
    print(f"Step 1: Extracting fingerprint for Dataset {FINETUNE_DATASET_ID}...")
    api.extract_fingerprint(
        finetune_dataset_id=FINETUNE_DATASET_ID
    )
    print("-> Fingerprint extracted.\n")


    # --- Step 2: Load and Apply the Pre-trained Model's Plans ---
    # This is the key step for aligning architectures. It takes the plans
    # (network topology, patch size, normalization, etc.) from the pre-trained
    # model and applies them to your new dataset's fingerprint, creating a new
    # plan file specifically for fine-tuning.
    # This wraps: nnUNetv2_move_plans_between_datasets CLI
    print(f"Step 2: Applying plans from Dataset {PRETRAINED_DATASET_ID} to Dataset {FINETUNE_DATASET_ID}...")
    api.apply_pretrained_plans(
        pretrained_dataset_id=PRETRAINED_DATASET_ID,
        finetune_dataset_id=FINETUNE_DATASET_ID,
        finetune_plans_identifier=FINETUNE_PLANS_ID
    )
    print(f"-> Plans applied. New plans identifier is '{FINETUNE_PLANS_ID}'.\n")

    # # ## copy data.json from raw to preprocessed folder
    src_data_json = os.path.join(nnUNet_raw, DST_DATA_NAME, 'dataset.json')
    dst_data_json = os.path.join(nnUNet_preprocessed, DST_DATA_NAME, 'dataset.json')
    shutil.copy(src_data_json, dst_data_json)


    # --- Step 3: Preprocess Your New Dataset ---
    # Now that your fine-tuning dataset has a valid (and aligned) plan file,
    # you can run the standard preprocessing pipeline on it. This will resample,
    # crop, and normalize your images according to the new plan.
    # This wraps: nnUNetv2_preprocess CLI
    print(f"Step 3: Preprocessing Dataset {FINETUNE_DATASET_ID} with the new plans...")
    api.preprocess_dataset(
        dataset_id=FINETUNE_DATASET_ID,
        plans_identifier=FINETUNE_PLANS_ID
    )
    print("-> Preprocessing complete.\n")

    # --- Step 4: Run the Fine-Tuning Training ---
    # Finally, you can start the training process. This function works just like the
    # standard `train` method but includes the `pretrained_weights` argument.
    # nnU-Net will load the weights from the specified checkpoint file before starting
    # the training, effectively fine-tuning the model on your new data.
    # This wraps: nnUNetv2_train -pretrained_weights CLI
    print(f"Step 4: Starting fine-tuning on Dataset {FINETUNE_DATASET_ID}...")
    api.finetune(
        finetune_dataset_id=FINETUNE_DATASET_ID,
        configuration='3d_fullres',
        fold=FOLD,
        pretrained_checkpoint_path=PRETRAINED_CHECKPOINT_PATH,
        plans_identifier=FINETUNE_PLANS_ID,
        num_epochs=N_EPOCHS,
        initial_lr=INIT_LR
    )
    print("\nFine-tuning process has been started!")


    # --- Step 5: Run the inference of the newly fine tunned model ---
    # This wraps:
    print(f"Step 5: Infering fine-tuned model on a testing dataset")
    api.predict(
        input_folder=INPUT_FOLDER_INFER,
        output_folder=OUTPUT_FOLDER_INFER_FINETUNE,
        dataset_name_or_id=DST_DATA_NAME,
        plans_identifier=FINETUNE_PLANS_ID,
        configuration='3d_fullres',
        folds=[FOLD]
    )


    ## --- Step 5+: Run the inference of the original pretrained model ---
    ## This wraps:
    print(f"Step 5: Infering pretrained model on a testing dataset")
    api.predict(
        input_folder=INPUT_FOLDER_INFER,
        output_folder=OUTPUT_FOLDER_INFER_PRETRAINED,
        dataset_name_or_id="Dataset770_BraTSGLIPreCropRegion",
        plans_identifier="nnUNetResEncUNetPlans",
        configuration='3d_fullres',
        folds=[FOLD]
    )