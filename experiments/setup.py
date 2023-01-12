from denoising_diffusion_pytorch import Unet, GaussianDiffusionSegmentationMapping, TrainerSegmentation
from experiments.config import TrainingConfig

def setup_trainer(config: TrainingConfig):
    model = Unet(
        dim=config.dim,
        dim_mults=config.dim_mults
    ).cuda()

    diffusion = GaussianDiffusionSegmentationMapping(
        model,
        image_size=config.image_size,
        margin=config.margin,
        regularization_margin=config.regularization_margin,
        regularize_to_white_image=config.regularize_to_white_image,
        loss_type=config.loss_type,
        timesteps=config.timesteps,           
        sampling_timesteps=config.sampling_timesteps,
        noising_timesteps=config.noising_timesteps,
        ddim_sampling_eta=config.ddim_sampling_eta,
        is_loss_time_dependent=config.is_loss_time_dependent
    ).cuda()

    trainer = TrainerSegmentation(
        diffusion,
        config.images_folder,
        config.segmentation_folder,
        augment_horizontal_flip=False,
        results_folder=config.results_folder,
        validate_every=config.validate_every,
        save_every=config.save_every,
        data_split=config.data_split,
        num_samples=config.num_samples,
        train_batch_size=config.train_batch_size,
        optimizer=config.optimizer,
        adam_betas=config.adam_betas,
        lr_decay=config.lr_decay,
        weight_decay=config.weight_decay,
        rms_prop_alpha=config.rms_prop_alpha,
        momentum=config.momentum,
        etas=config.etas,
        step_sizes=config.step_sizes,
        train_lr=config.train_lr,
        train_num_steps=config.train_num_steps,    
        gradient_accumulate_every=config.gradient_accumulate_every,
        ema_decay=config.ema_decay,
        amp=True
    )

    return trainer