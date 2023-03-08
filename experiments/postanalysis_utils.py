from pathlib import Path
from functools import reduce
from typing import List, Optional, Mapping
import re
import matplotlib.pyplot as plt

import pandas as pd

from experiments.config import PathLike


def get_last_checkpoint(model_folder: PathLike):
    model_folder = Path(model_folder)
    models = model_folder.rglob("model-*.pt")
    regex = re.compile("model-(.*).pt")
    models_checkpoints = [int(regex.search(str(path)).group(1)) for path in models]
    return max(models_checkpoints)


def extract_loss_function_cross_fold(
    model_folder_path: PathLike,
    graph_title: str,
    fold_folders_names: Optional[List[str]] = None,
    last_checkpoints: Optional[Mapping[str, int]] = None,
    k: int = 5
):
    model_folder_path = Path(model_folder_path)

    if fold_folders_names is None:
        fold_folders_names = [f"fold_{i}" for i in range(k)]
    if last_checkpoints is None:
        last_checkpoints = {fold: get_last_checkpoint(model_folder_path / fold) for fold in fold_folders_names}

    loss_dfs = [pd.read_csv(model_folder_path / fold / f"validation_loss-{checkpoint}.csv")
                    for fold, checkpoint in last_checkpoints.items()]

    for fold_name, loss_df in zip(fold_folders_names, loss_dfs):
        loss_df.index.name = fold_name

    full_df = reduce(
        lambda left, right: pd.merge(left, right,
                                     on="epoch", how="outer", suffixes=(None, f"_{right.index.name}")),
        [pd.DataFrame(columns=("epoch", "loss"))] + loss_dfs,
    )
    full_df.to_csv(model_folder_path / "validation_loss.csv")

    full_df.rename(columns={"epoch": "Training iteration"}, inplace=True)
    num_folds = len(fold_folders_names)
    ax = plt.subplot(xlabel="Training iteration", ylabel="Loss value", title=graph_title)
    for fold in fold_folders_names:
        full_df.plot("Training iteration", f"loss_{fold}", ax=ax)

    plt.savefig(model_folder_path / "loss_function.png")
    plt.close()


def main():
    result_paths = {
        "/data/pg-organoid_data/segmentation_ddpm/results/experiment_50_samples": "5-fold validation loss descent with\ntriplet loss for 50 samples",
        "/data/pg-organoid_data/segmentation_ddpm/results/experiment_100_samples": "5-fold validation loss descent with\ntriplet loss for 100 samples",
        "/data/pg-organoid_data/segmentation_ddpm/results/experiment_200_samples": "5-fold validation loss descent with\ntriplet loss for 200 samples",
        "/data/pg-organoid_data/segmentation_ddpm/results/experiment_500_samples": "5-fold validation loss descent with\ntriplet loss for 500 samples",
        "/data/pg-organoid_data/segmentation_ddpm/results/experiment_mse_50_samples": "5-fold validation loss descent with\nMSE for 50 samples",
        "/data/pg-organoid_data/segmentation_ddpm/results/experiment_mse_100_samples": "5-fold validation loss descent with\nMSE for 100 samples",
        "/data/pg-organoid_data/segmentation_ddpm/results/experiment_mse_200_samples": "5-fold validation loss descent with\nMSE for 200 samples",
        "/data/pg-organoid_data/segmentation_ddpm/results/experiment_mse": "5-fold validation loss descent with\nMSE for 500 samples",
        "/data/pg-organoid_data/segmentation_ddpm/results/experiment_triplet_4": "5-fold validation loss descent with\na margin value of 4.0",
        "/data/pg-organoid_data/segmentation_ddpm/results/experiment_triplet_8": "5-fold validation loss descent with\na margin value of 8.0",
        "/data/pg-organoid_data/segmentation_ddpm/results/experiment_triplet_24": "5-fold validation loss descent with\na margin value of 24.0"
    }
    [extract_loss_function_cross_fold(model_path, graph_title) for model_path, graph_title in result_paths.items()]


if __name__ == "__main__":
    main()
