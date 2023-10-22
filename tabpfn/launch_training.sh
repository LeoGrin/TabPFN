#!/bin/bash
#SBATCH --time=45:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=parietal
#SBATCH --gpus=1

bash -l -c "micromamba activate tabpfn;python train_prior_custom_copy.py --save_every 10 --p_categorical 0.15 --correlation_proba_min 0.0 --correlation_proba_max 0.2 --batch_size 32 --max_depth_lambda 0.35 --n_estimators_lambda 0.15 --prior trees --correlation_strength_min 0.0 --correlation_strength_max 0.2 --min_categories 2 --max_categories 10 --random_feature_removal 0.2 --wandb --local_rank 0"
