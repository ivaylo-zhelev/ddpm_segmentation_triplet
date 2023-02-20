module load Python/3.8.6-GCCcore-10.2.0

cd /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet
git pull

source /data/s3782255/.envs/diffusion_segm/bin/activate
python -m pip install /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/
pip uninstall denoising_diffusion_triplet
pip install .

sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/jobs/experiment_1.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/jobs/experiment_2.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/jobs/experiment_3.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/jobs/experiment_4.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/jobs/experiment_5.sh