import os
import torch
import torch.utils.data
import numpy as np
from typing import Tuple
from termcolor import cprint

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", device: str = 'cpu') -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.device = device

        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).to(self.device)
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt")).to(self.device)
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt")).to(self.device)
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        # 正規化と白色化
        self.normalize()
        self.whiten()
        
    def normalize(self) -> None:
        self.mean = torch.mean(self.X, dim=(0, 2), keepdim=True).to(self.device)
        self.std = torch.std(self.X, dim=(0, 2), keepdim=True).to(self.device)
        self.X = (self.X - self.mean) / self.std

    def whiten(self) -> None:
        X_flat = self.X.view(self.X.size(0), -1).to(self.device)
        cov = torch.matmul(X_flat.T, X_flat) / X_flat.size(0)
        U, S, V = torch.svd(cov)
        self.whitening_matrix = torch.matmul(U, torch.diag(1.0 / torch.sqrt(S + 1e-5))).to(self.device)
        self.X = torch.matmul(X_flat, self.whitening_matrix).view(self.X.size()).to(self.device)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

# データセットをGPUで使用する例
train_dataset = ThingsMEGDataset(split="train", data_dir="data", device='cuda')
val_dataset = ThingsMEGDataset(split="val", data_dir="data", device='cuda')
