#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00
#SBATCH --mem=8000
#SBATCH --job-name=ddpm_segmentation_experiment_5
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=i.zhelev@student.rug.nl
#SBATCH --output=job-ddpm_segmentation_experiment_5.log


module purge
module load Python/3.8.6-GCCcore-10.2.0
module load CUDA

source /data/s3782255/.envs/diffusion/bin/activate

python /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/experiments/training_script.py --config-file=/home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/config_5.yaml