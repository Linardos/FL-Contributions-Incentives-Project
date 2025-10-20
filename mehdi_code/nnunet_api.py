import sys
import torch
from torch.backends import cudnn
from nnunetv2.run.run_training import run_training
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint
from batchgenerators.utilities.file_and_folder_operations import join, subfiles, load_json
from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import extract_fingerprint_entry, preprocess_entry
from nnunetv2.experiment_planning.plans_for_pretraining.move_plans_between_datasets import entry_point_move_plans_between_datasets

class NnUnetApi:
    def train(self, dataset_name_or_id, configuration, fold, trainer_class_name='nnUNetTrainer', plans_identifier='nnUNetPlans', device=torch.device('cuda')):
        # This function is a direct Python entry point, so we can call it normally.
        run_training(str(dataset_name_or_id), configuration, fold, trainer_class_name, plans_identifier, requested_device=device)

    def predict(self, input_folder, output_folder, dataset_name_or_id, configuration, folds=(0,), checkpoint_name='checkpoint_final.pth', plans_identifier='nnUNetPlans'):
        # This is a class, so we instantiate and use it.
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda', 0),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        
        # The model folder path is constructed using the plans_identifier.
        model_folder = join(nnUNet_results, str(dataset_name_or_id), f'nnUNetTrainer__{plans_identifier}__{configuration}')
        
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=folds,
            checkpoint_name=checkpoint_name,
        )
        
        # Create a list of lists, where each sub-list contains all modalities of a single case
        input_files = subfiles(input_folder, suffix=predictor.dataset_json['file_ending'], join=False, sort=True)
        case_identifiers = sorted(list(set([i[:-len(predictor.dataset_json['file_ending']) - 5] for i in input_files])))
        list_of_lists_of_files = []
        for c in case_identifiers:
            files_for_case = [join(input_folder, i) for i in input_files if i.startswith(c)]
            list_of_lists_of_files.append(files_for_case)

        print(f"Found {len(list_of_lists_of_files)} cases to predict")

        predictor.predict_from_files(
            list_of_lists_of_files,
            output_folder,
            save_probabilities=False,
            overwrite=False,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )

    def load_pretrained_plan(self, pretrained_dataset_name_or_id, plans_identifier='nnUNetPlans'):
        plans_file = join(nnUNet_preprocessed, str(pretrained_dataset_name_or_id), plans_identifier + '.json')
        return load_json(plans_file)

    def extract_fingerprint(self, finetune_dataset_id):
        # This is a CLI entry point, so we need to simulate a command-line call.
        original_argv = sys.argv
        try:
            # Simulates: nnUNetv2_extract_fingerprint -d DATASET_ID
            sys.argv = ['', '-d', str(finetune_dataset_id)]
            extract_fingerprint_entry()
        finally:
            # Restore original arguments
            sys.argv = original_argv

    def apply_pretrained_plans(self, pretrained_dataset_id, finetune_dataset_id, pretrained_plans_identifier='nnUNetResEncUNetPlans', finetune_plans_identifier='nnUNetPlans_finetune'):
        # This is a CLI entry point, so we need to simulate a command-line call.
        original_argv = sys.argv
        try:
            # Simulates: nnUNetv2_move_plans_between_datasets -s SOURCE_ID -t TARGET_ID -sp SOURCE_PLANS -tp TARGET_PLANS
            sys.argv = ['', '-s', str(pretrained_dataset_id), '-t', str(finetune_dataset_id), '-sp', pretrained_plans_identifier, '-tp', finetune_plans_identifier]
            entry_point_move_plans_between_datasets()
        finally:
            sys.argv = original_argv

    def preprocess_dataset(self, dataset_id, plans_identifier, configurations=('3d_fullres',)):
        # This is a CLI entry point, so we need to simulate a command-line call.
        original_argv = sys.argv
        try:
            # Simulates: nnUNetv2_preprocess -d DATASET_ID -plans_name PLANS_NAME -c CONFIGS
            args = ['-d', str(dataset_id), '-plans_name', plans_identifier, '-c'] + list(configurations)
            sys.argv = [''] + args
            preprocess_entry()
        finally:
            sys.argv = original_argv

    def finetune(self, finetune_dataset_id, configuration, fold, pretrained_checkpoint_path,
                 plans_identifier='nnUNetPlans_finetune', trainer_class_name='nnUNetTrainer',
                 num_epochs: int = 1000, initial_lr: float = 1e-2, device=torch.device('cuda')):
        """
        Run fine-tuning with custom epoch and learning rate settings.
        This method replicates the logic of `run_training` to allow for hyperparameter modification
        without changing the core `run_training.py` script.
        """
        # Get the trainer instance
        nnunet_trainer = get_trainer_from_args(str(finetune_dataset_id), configuration, fold, trainer_class_name,
                                               plans_identifier, device=device)

        # Set custom hyperparameters before initialization and weight loading
        nnunet_trainer.num_epochs = num_epochs
        nnunet_trainer.initial_lr = initial_lr

        # Load pretrained weights
        maybe_load_checkpoint(nnunet_trainer, continue_training=False, validation_only=False,
                              pretrained_weights_file=pretrained_checkpoint_path)

        # Set up cuDNN and run training
        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        nnunet_trainer.run_training()
