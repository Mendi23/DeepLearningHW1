import math

import numpy as np
import torch
import torch.utils.data.sampler as Sampler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

from .datasets import SubsetDataset



def create_train_validation_loaders(dataset: Dataset, validation_ratio,
                                    batch_size=100, num_workers=1):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not(0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    # TODO: Create two DataLoader instances, dl_train and dl_valid.
    # They should together represent a train/validation split of the given
    # dataset. Make sure that:
    # 1. Validation set size is validation_ratio * total number of samples.
    # 2. No sample is in both datasets. You can select samples at random
    #    from the dataset.

    # ====== YOUR CODE: ======
    #Split is done randomely! (And Sample is also done randomely again -doesnt really matter)
    
    train_size = int(len(dataset)* (1 - validation_ratio))
    
    perm = torch.randperm(len(dataset)).tolist()
    train_perm = perm[:train_size]
    validation_perm = perm[train_size:]
    
    train_sampler = SubsetRandomSampler(train_perm)
    validation_sampler = SubsetRandomSampler(validation_perm)
    
    train_dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           num_workers=num_workers, sampler=train_sampler)
    valid_dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           num_workers=num_workers, sampler=validation_sampler)
    
    # ========================

    return train_dl, valid_dl

