#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1-00:00
#SBATCH --mem=10000
#SBATCH --job-name=ddpm_sampling_experiment_2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=i.zhelev@student.rug.nl
#SBATCH --output=job-ddpm_sampling_experiment_2.log


module purge
module load Python/3.8.6-GCCcore-10.2.0
module load CUDA
module load Boost

source /data/s3782255/.envs/diffusion_segm/bin/activate
python -m pip install /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/

python /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/experiments/training_script.py --training-config-file=/home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/config_2.yaml --sampling-config-file=/home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/sampling_config.yaml