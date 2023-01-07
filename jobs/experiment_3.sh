#!/bin/bash
#SBATCH --nodes=4
#SBATCH --time=3-00:00
#SBATCH --mem=4000
#SBATCH --job-name=ddpm_segmentation_experiment_3
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=i.zhelev@student.rug.nl
#SBATCH --output=job-ddpm_segmentation_experiment_3.log


module purge
module load Python/3.6.4-foss-2018a
module load CUDA/9.1.85
module load Boost/1.66.0-foss-2018a-Python-3.6.4

source /data/s3782255/.envs/diffusion/bin/activate

python /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/experiments/training_script.py --config-file=/home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/configs/experiment_3.yaml