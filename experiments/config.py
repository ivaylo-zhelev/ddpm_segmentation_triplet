from typing import Tuple, Union, Optional
from pathlib import Path
from dataclasses import dataclass, fields, _MISSING_TYPE


@dataclass
class TrainingConfig:
    images_folder: Union[str, Path]
    segmentation_folder: Union[str, Path]
    results_folder: Union[str, Path]

    dim: int = 64
    dim_mults: Tuple[int, int, int, int] = (1, 2, 4, 8)

    image_size: int = 320
    margin: float = 1.0
    regularization_margin: float = 10.0
    regularize_to_white_image: bool = True
    loss_type: str = "regularized_triplet"
    timesteps: int = 1000           
    sampling_timesteps: int = 100
    noising_timesteps: Optional[int] = None
    ddim_sampling_eta: float = 0.0
    is_loss_time_dependent: bool = False

    optimizer: str = "adam"
    adam_betas: Tuple[float, float] = (0.9, 0.99)
    lr_decay: float = 0
    weight_decay: float = 0
    rms_prop_alpha: float = 0.99
    momentum: float = 0
    etas: Tuple[float, float] = (0.5, 1.2)
    step_sizes: Tuple[float, float] = (1e-06, 50)

    validate_every: int = 2000
    save_every: int = 2000
    load_milestone: int = 1
    data_split: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    num_samples: int = 25
    train_batch_size: int = 8
    train_lr: float = 8e-5
    train_num_steps: int = 100000    
    gradient_accumulate_every: int = 2
    ema_decay: float = 0.995

    def __post_init__(self):
        # Loop through the fields
        for field in fields(self):
            # If there is a default and the value of the field is none we can assign a value
            if not isinstance(field.default, _MISSING_TYPE) and getattr(self, field.name) is None:
                setattr(self, field.name, field.default)

        self.images_folder = Path(self.images_folder)
        self.segmentation_folder = Path(self.segmentation_folder)
        self.results_folder = Path(self.results_folder)
