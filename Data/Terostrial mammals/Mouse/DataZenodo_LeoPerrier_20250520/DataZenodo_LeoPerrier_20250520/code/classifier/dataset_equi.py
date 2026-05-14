# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:30:40 2023

@author: adminlocal
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict


def create_datasets(path, splits, seq_len):
    
    
    def min_images(folder_path):
        image_counts = defaultdict(int)

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.png'):
                    image_counts[root] += 1

        if image_counts:
            min_count = min(image_counts.values())
            return min_count
        else:
            return 0
        
        
    datasets = {"train": [], "val": [], "test": []}
    subjects = os.listdir(path)

    for label, subject in enumerate(subjects):
        list_spectro = np.array(os.listdir(f"{path}/{subject}"))
        list_spectro_equi = np.random.choice(list_spectro, size=min_images(path), replace=False)
        tmp = np.arange(0, len(list_spectro_equi)//seq_len*seq_len, seq_len)
        np.random.shuffle(tmp)
        mask = (np.cumsum(splits)/100*len(tmp)).astype(int)
        sets = np.split(tmp, mask[:-1])

        for i, (key, val) in enumerate(datasets.items()):
            spectro = list_spectro[sets[i]]
            spectro = [f"{path}/{subject}/{name}" for name in spectro]
            labels = [label] * len(spectro)
            datasets[key] += zip(spectro, labels)

    nb_subjects = len(subjects)
    datasets["label2subject"] = dict(zip(range(nb_subjects), subjects))
    datasets["nb_subjects"] = nb_subjects

    return datasets



class SpectrogramDataset(Dataset):
    def __init__(self, dataset, seq_len, transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        last_valid_path, label = self.dataset[idx]
        spectrogram = np.asarray(Image.open(last_valid_path))[..., np.newaxis]

        for i in range(self.seq_len - 1):
            next_path = last_valid_path.split("_")
            next_path[-2] = str(int(next_path[-2]) + i + 1)
            next_path = "_".join(next_path)
            try:
                next_spectro = np.asarray(Image.open(next_path))[..., np.newaxis]
                last_valid_path = next_path
            except FileNotFoundError:
                next_spectro = np.asarray(Image.open(last_valid_path))[..., np.newaxis]

            spectrogram = np.concatenate((spectrogram, next_spectro), axis=2)

        if self.transform:
            spectrogram = self.transform(np.array(spectrogram))

        label = torch.tensor(label, dtype=torch.long)
        return spectrogram, label
