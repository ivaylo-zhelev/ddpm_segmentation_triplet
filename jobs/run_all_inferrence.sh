#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1-00:00
#SBATCH --mem=10000
#SBATCH --job-name=all_inferrence
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=i.zhelev@student.rug.nl
#SBATCH --output=job-all_inferrence.log


module purge
module load Python/3.8.6-GCCcore-10.2.0
module load CUDA
module load Boost

source /data/s3782255/.envs/diffusion_segm/bin/activate

python /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/experiments/inferrence_script.py -c /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/config_50_samples.yaml
python /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/experiments/inferrence_script.py -c /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/config_100_samples.yaml
python /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/experiments/inferrence_script.py -c /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/config_200_samples.yaml
python /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/experiments/inferrence_script.py -c /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/config_500_samples.yaml
python /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/experiments/inferrence_script.py -c /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/config_mse_50_samples.yaml
python /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/experiments/inferrence_script.py -c /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/config_mse_100_samples.yaml
python /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/experiments/inferrence_script.py -c /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/config_mse_200_samples.yaml
python /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/experiments/inferrence_script.py -c /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/config_mse.yaml
python /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/experiments/inferrence_script.py -c /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/config_triplet_4.yaml
python /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/experiments/inferrence_script.py -c /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/config_triplet_8.yaml
python /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/experiments/inferrence_script.py -c /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/config_triplet_24.yaml