# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import json
import numpy as np
import logging
import os
import random
import scipy.io
import scipy.ndimage
import sys
import torch.utils.data.dataset

from enum import Enum, unique

import utils.binvox_rw


@unique
class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


# //////////////////////////////// = End of DatasetType Class Definition = ///////////////////////////////// #


class ShapeNetDataset(torch.utils.data.dataset.Dataset):
    """ShapeNetDataset class used for PyTorch DataLoader"""
    def __init__(self, dataset_type, file_list):
        self.dataset_type = dataset_type
        self.file_list = file_list
        #self.n_views_rendering = n_views_rendering

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        volume, c_space = self.get_datum(idx)
        np_vox = np.zeros([1, 32, 32, 32])
        np_vox[0, :, :, :] = volume
        np_vox = torch.tensor(np_vox).float()
        return np_vox, c_space


    def get_datum(self, idx):
        volume_path = self.file_list[idx]['volume']
        c_space_path = self.file_list[idx]['c_space']

        # Get data of c_space
        c_space = np.load(c_space_path)
        c_space = torch.from_numpy(c_space.astype(np.float32)).clone()

        # Get data of volume
        _, suffix = os.path.splitext(volume_path)

        if suffix == '.mat':
            volume = scipy.io.loadmat(volume_path)
            volume = volume['Volume'].astype(np.float32)
        elif suffix == '.binvox':
            with open(volume_path, 'rb') as f:
                volume = utils.binvox_rw.read_as_3d_array(f, fix_coords=False)
                volume = volume.data.astype(np.float32)


        return volume, c_space


# //////////////////////////////// = End of ShapeNetDataset Class Definition = ///////////////////////////////// #


class VoxCspaceDataLoader:
    def __init__(self, cfg):
        self.volume_path_template = cfg.DATASETS.VOX_CSPACE.VOXEL_PATH
        self.c_space_path_template = cfg.DATASETS.VOX_CSPACE.CSPACE_PATH 

        # Load all taxonomies of the dataset
        with open(cfg.DATASETS.VOX_CSPACE.JSON_PATH, encoding='utf-8') as file:
            self.datasets = json.loads(file.read())
        

    def get_dataset(self, dataset_type):
        files = []

        # Load data
        for data in self.datasets:
            samples = []
            if dataset_type == DatasetType.TRAIN:
                samples = data['train']
            elif dataset_type == DatasetType.TEST:
                samples = data['test']
            elif dataset_type == DatasetType.VAL:
                samples = data['val']

            files.extend(self.get_files_of_data(samples))

        logging.info('Complete collecting files of the dataset. Total files: %d.' % (len(files)))
        return ShapeNetDataset(dataset_type, files)

    def get_files_of_data(self, samples):
        files_of_data = []

        for sample_idx, sample_name in enumerate(samples):
            # Get file path of cspaces
            c_space_file_path = self.c_space_path_template % (sample_name)
            if not os.path.exists(c_space_file_path):
                logging.warn('Ignore sample %s since volume file not exists.' % (sample_name))
                continue

            # Get file list of volumes
            volume_file_path = self.volume_path_template % (sample_name)

            files_of_data.append({
                'sample_name': sample_name,
                'volume': volume_file_path,
                'c_space': c_space_file_path,
            })
        return files_of_data


# /////////////////////////////// = End of ShapeNetDataLoader Class Definition = /////////////////////////////// #



DATASET_LOADER_MAPPING = {
    'VoxCspace': VoxCspaceDataLoader,
}  # yapf: disable