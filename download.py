# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained DiT models
"""
from torchvision.datasets.utils import download_url
import torch
import os


pretrained_models = {'DiT-XL-2-256x256.pt', 'DiT-XL-2-512x512.pt'}     # 'DiT-XL-2-512x512.pt',


def find_model(model_name):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """

    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint


if __name__ == "__main__":

    print('Done.')
