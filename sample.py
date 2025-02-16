# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch

from diffusion.dataset import WindDataDataset, wind_data, wind_data_zone2, WindDataDataset_zone2

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models
import argparse
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/1", "Only DiT-XL/1 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    model = DiT_models[args.model](
        input_size=args.seq_len,
        exo_channels=10 if args.partial else 20,
        cross_attn=False,
    ).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))

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
    print(f"{df_x_LS.shape}, {df_y_LS.shape}")
    print('all zones #LS %s days #VS %s days # TEST %s days' % (nb_days_LS, nb_days_VS, nb_days_TEST))
    if args.partial:
        train_dataset_LS = WindDataDataset_zone2(df_y_LS, df_x_LS)
        train_dataset_VS = WindDataDataset_zone2(df_y_VS, df_x_VS, train_dataset_LS.scaler)
        test_dataset = WindDataDataset_zone2(df_y_TEST, df_x_TEST, train_dataset_LS.scaler)
    else:
        train_dataset_LS = WindDataDataset(df_y_LS, df_x_LS)
        train_dataset_VS = WindDataDataset(df_y_VS, df_x_VS, train_dataset_LS.scaler)
        test_dataset = WindDataDataset(df_y_TEST, df_x_TEST, train_dataset_LS.scaler)


    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"len(test_loader): {len(test_loader)}")

    # Evaluate model on test set:
    sample_repeats = 100  # 每个样本采样次数
    all_predictions = []  # 保存所有样本的采样结果
    true_values = []  # 保存所有样本的真实值
    using_cfg = args.cfg_scale > 1.0
    start_time = time.time()
    with torch.no_grad():
        i = 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            _x = x.repeat(sample_repeats, 1, 1)
            y = y.repeat(sample_repeats, 1, 1)
            z = torch.randn_like(_x, device=device)

            z = torch.cat([z, z], 0)

            y_null = torch.randn_like(y, device=device)
            _y = torch.cat([y, y_null], 0)
            model_kwargs = dict(exo_c=_y, cfg_scale=args.cfg_scale)
            # model_kwargs = dict(y=_y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg


            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                device=device
            )

            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples (B, 1, seq)
            samples = samples.squeeze(1).cpu().numpy()  # shape: (sample_repeats, seq_len)
            all_predictions.append(samples)

            print(samples.shape, x.shape)

            x = x.cpu().numpy().squeeze(0)  # shape: (seq_len,)
            true_values.append(x)


    all_time = time.time() - start_time
    all_predictions = np.array(all_predictions)  # shape: (num_samples, sample_repeats, seq_len)
    all_predictions = all_predictions.transpose(0, 2, 1) # shape: (num_samples, seq_len, sample_repeats)
    true_values = np.concatenate(true_values, axis=0)  # shape: (num_samples, seq_len)
    print(f"all_predictions.shape: {all_predictions.shape}, true_values.shape: {true_values.shape}")
    np.save("pred_wind.npy", all_predictions)
    np.save("true_wind.npy", true_values)
    print(f"all_time: {all_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/1")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/1 model).")
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
