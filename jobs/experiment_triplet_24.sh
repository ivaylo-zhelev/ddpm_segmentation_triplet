#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1-00:00
#SBATCH --mem=10000
#SBATCH --job-name=ddpm_segmentation_experiment_triplet_24
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=i.zhelev@student.rug.nl
#SBATCH --output=job-ddpm_segmentation_experiment_triplet_24.log


module purge
module load Python/3.8.6-GCCcore-10.2.0
module load CUDA
module load Boost

source /data/s3782255/.envs/diffusion_segm/bin/activate

python /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/experiments/run_k_fold.py -t /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/config_triplet_24.yaml -k /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/k_fold_config.yaml