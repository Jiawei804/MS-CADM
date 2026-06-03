# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import math
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from diffusion.dataset import wind_data, WindDataDataset, wind_data_zone2, WindDataDataset_zone2

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import datetime

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."



    device = torch.cuda.current_device()
    seed = args.global_seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{experiment_dir}/log.txt")]
        )
    logger = logging.getLogger(__name__)
    logger.info(f"Experiment directory created at {experiment_dir}")


    # Create model:
    model = DiT_models[args.model](
        input_size=args.seq_len,
        exo_channels=10 if args.partial else 20,
        cross_attn=False,
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    model = model.to(device)
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)


    # prepare data
    if args.partial:
        data = wind_data_zone2(path_name=args.data_path, test_size=50, random_state=0)
    else:
        data = wind_data(path_name=args.data_path, test_size=50, random_state=0)

    df_x_LS = data[0].copy()
    df_y_LS = data[1].copy()
    df_x_VS = data[2].copy()
    df_y_VS = data[3].copy()
    df_x_TEST = data[4].copy()
    df_y_TEST = data[5].copy()

    nb_days_LS = len(df_y_LS)
    nb_days_VS = len(df_y_VS)
    nb_days_TEST = len(df_y_TEST)
    logger.info(f"{df_x_LS.shape}, {df_y_LS.shape}")
    logger.info('all zones #LS %s days #VS %s days # TEST %s days' % (nb_days_LS, nb_days_VS, nb_days_TEST))

    if args.partial:
        train_dataset_LS = WindDataDataset_zone2(df_y_LS, df_x_LS)
        train_dataset_VS = WindDataDataset_zone2(df_y_VS, df_x_VS, train_dataset_LS.scaler)
        test_dataset = WindDataDataset_zone2(df_y_TEST, df_x_TEST, train_dataset_LS.scaler)
    else:
        train_dataset_LS = WindDataDataset(df_y_LS, df_x_LS)
        train_dataset_VS = WindDataDataset(df_y_VS, df_x_VS, train_dataset_LS.scaler)
        test_dataset = WindDataDataset(df_y_TEST, df_x_TEST, train_dataset_LS.scaler)

    train_loader = DataLoader(
        train_dataset_LS,
        batch_size=args.global_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        train_dataset_VS,
        batch_size=int(len(train_dataset_LS)*0.1),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(f"Dataset contains {len(train_dataset_LS):,} datas ({args.data_path})")


    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    if args.max_train_steps is not None:
        args.epochs = args.max_train_steps // len(train_loader) + 1
    else:
        args.max_train_steps = args.epochs * len(train_loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    train_loss_list = []
    ll_VS_min = float('inf')
    logger.info(f"Training for {args.epochs} epochs...  (max steps: {args.max_train_steps})")
    for epoch in range(args.epochs):

        # logger.info(f"Beginning epoch {epoch}...")
        train_loss = 0
        for x, c in train_loader:
            x = x.to(device)
            c = c.to(device)


            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(exo_c=c)
            # model_kwargs = dict(y=c)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            train_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # avg_loss:
                avg_loss = running_loss / log_steps
                train_loss_list.append(avg_loss)
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

                # Evaluate model on validation set:

                val_loss = 0
                model.eval()
                with torch.no_grad():
                    for x, c in val_loader:
                        x = x.to(device)
                        c = c.to(device)

                        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                        model_kwargs = dict(exo_c=c)
                        # model_kwargs = dict(y=c)
                        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                        loss = loss_dict["loss"].mean()
                        # logger.info(f"Validation Loss: {loss.item():.4f}")
                        val_loss += loss.item()
                    logger.info(f"Validation Loss: {val_loss / len(val_loader):.4f}")


                if not math.isnan(val_loss) and val_loss <= ll_VS_min and train_steps > 20000:
                    checkpoint = {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/best_model.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved best model!!!")
                    ll_VS_min = val_loss
                model.eval()
            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:

                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")



    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...


    logger.info("Done!")



if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/1")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    main(args)
