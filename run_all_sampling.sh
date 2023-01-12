module load Python/3.8.6-GCCcore-10.2.0

cd /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet
git pull

source /data/s3782255/.envs/diffusion_segm/bin/activate
pip uninstall denoising_diffusion_triplet
pip install .

sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/ddim_sampling/experiment_3/job_2.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/ddim_sampling/experiment_3/job_1.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/ddim_sampling/experiment_3/job_3.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/ddim_sampling/experiment_2/job_2.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/ddim_sampling/experiment_2/job_1.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/ddim_sampling/experiment_2/job_3.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/ddim_sampling/experiment_1/job_2.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/ddim_sampling/experiment_1/job_1.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/ddim_sampling/experiment_1/job_3.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/experiment_3/job_2.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/experiment_3/job_1.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/experiment_3/job_3.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/experiment_3/job_5.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/experiment_3/job_4.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/experiment_2/job_2.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/experiment_2/job_1.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/experiment_2/job_3.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/experiment_2/job_5.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/experiment_2/job_4.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/experiment_1/job_2.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/experiment_1/job_1.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/experiment_1/job_3.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/experiment_1/job_5.sh
sbatch /home/s3782255/segmentation_ddpm/ddpm_segmentation_triplet/sampling_jobs/experiment_1/job_4.sh