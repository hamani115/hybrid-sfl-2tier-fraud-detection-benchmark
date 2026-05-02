import os
from functools import partial

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from flwr_datasets import FederatedDataset
from sklearn.preprocessing import StandardScaler

from src.utils.environment_variables import EnvironmentVariables as EV


class DictTensorDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "label": self.y[idx],
        }


def _apply_transforms(batch, transforms):
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch


def _dirichlet_label_partition(y, num_partitions, alpha, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    client_indices = [[] for _ in range(num_partitions)]

    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)

        proportions = rng.dirichlet(np.repeat(alpha, num_partitions))
        split_points = (np.cumsum(proportions) * len(label_indices)).astype(int)[:-1]
        splits = np.split(label_indices, split_points)

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    for client_id in range(num_partitions):
        rng.shuffle(client_indices[client_id])

    return client_indices


def _load_creditcard_dataset(dataset_cfg, partitioning_cfg, seed, client_idx):
    df = pd.read_csv(dataset_cfg.csv_path)

    # Chronological split is more realistic for fraud data.
    df = df.sort_values(dataset_cfg.time_col).reset_index(drop=True)

    n_train = int(len(df) * dataset_cfg.split.train_fraction)

    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()

    feature_cols = list(dataset_cfg.feature_cols)
    target_col = dataset_cfg.target_col

    x_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    x_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    num_partitions = int(partitioning_cfg.num_partitions)
    alpha = float(partitioning_cfg.alpha)

    partitions = _dirichlet_label_partition(
        y=y_train,
        num_partitions=num_partitions,
        alpha=alpha,
        seed=seed,
    )

    client_idx = int(client_idx)
    train_indices = partitions[client_idx]

    client_x_train = x_train[train_indices]
    client_y_train = y_train[train_indices]

    print(
        f"[creditcard] client={client_idx} "
        f"train_samples={len(client_y_train)} "
        f"class_counts={np.bincount(client_y_train, minlength=2).tolist()} "
        f"test_samples={len(y_test)} "
        f"test_counts={np.bincount(y_test, minlength=2).tolist()}"
    )

    return {
        "train": DictTensorDataset(client_x_train, client_y_train),
        "test": DictTensorDataset(x_test, y_test),
    }


def _load_flwr_image_dataset(dataset_cfg, partitioning_cfg, seed, client_idx):
    partitioner = instantiate(partitioning_cfg)
    fds = FederatedDataset(
        dataset=dataset_cfg.dataset_name,
        partitioners={"train": partitioner},
        seed=seed,
        cache_dir=os.getenv(EV.DATA_HOME_FOLDER, None),
    )

    dataset = fds.load_partition(client_idx)
    transforms = instantiate(dataset_cfg.transforms)

    _apply_transforms_partial = partial(_apply_transforms, transforms=transforms)
    dataset = dataset.with_transform(_apply_transforms_partial)
    dataset = dataset.train_test_split(test_size=dataset_cfg.test_percentage, seed=seed)

    return dataset


def get_dataset_from_cfg(dataset_cfg, partitioning_cfg, seed, client_idx):
    if dataset_cfg.dataset_name == "creditcard":
        return _load_creditcard_dataset(dataset_cfg, partitioning_cfg, seed, client_idx)

    return _load_flwr_image_dataset(dataset_cfg, partitioning_cfg, seed, client_idx)