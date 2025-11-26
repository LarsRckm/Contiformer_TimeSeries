# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import math
import random
import logging
import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
from torch import nn
import torch
import torchcde
from contiformer import AttrDict, EncoderLayer
from dataset_timeSeries import TimeSeriesDataset_Interpolation_roundedInput
from torch.utils.data import DataLoader
from tqdm import tqdm

# matplotlib.use('agg')


def get_logger(name):
    logger = logging.getLogger(name)
    filename = f'{name}.log'
    fh = logging.FileHandler(filename, mode='a+', encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=False)
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default='./train_dir/')
parser.add_argument('--val_dir_pictures', type=str, default='./val_pictures/')
parser.add_argument('--val_dir_data', type=str, default='./val_data/')
parser.add_argument('--model_name', type=str, default='Contiformer',
                    choices=['Neural_ODE', 'Contiformer'])
parser.add_argument('--log_step', type=int, default=50)
parser.add_argument('--seed', type=int, default=27)

## parameters for Contiformer
parser.add_argument('--atol', type=float, default=0.1)
parser.add_argument('--rtol', type=float, default=0.1)
parser.add_argument('--method', type=str, default='rk4')
parser.add_argument('--dropout', type=float, default=0.1)

##parameters for timeseries generation
parser.add_argument('--y_lim_low', type=int, default=10)
parser.add_argument('--y_lim_high', type=int, default=10000)
parser.add_argument('--train_count', type=int, default=10)
parser.add_argument('--val_count', type=int, default=1)
parser.add_argument('--number_x_values', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=10) #ausprobieren
parser.add_argument('--random_number_range_distribution', type=str, default="norm")
parser.add_argument('--random_number_range_mean', type=int, default=0)
parser.add_argument('--random_number_range_std', type=int, default=5)
parser.add_argument('--spline_value_low', type=int, default=800000)
parser.add_argument('--spline_value_high', type=int, default=1100000)
parser.add_argument('--vocab_size', type=int, default=100000)
parser.add_argument('--noise_std_distribution', type=str, default="norm")
parser.add_argument('--noise_std_mean', type=int, default=0)
parser.add_argument('--noise_std_std', type=int, default=0.15)
parser.add_argument('--interpolation_min_width', type=int, default=10)
parser.add_argument('--interpolation_max_width', type=int, default=100)
parser.add_argument('--interpolation_max_count', type=int, default=10)
parser.add_argument('--offset', type=int, default=10)
parser.add_argument('--x_lim_low', type=int, default=0)
parser.add_argument('--x_lim_high', type=int, default=1000)

args = parser.parse_args()

if not os.path.exists(args.train_dir):
    os.makedirs(args.train_dir)

if not os.path.exists(args.val_dir_pictures):
    os.makedirs(args.val_dir_pictures)

if not os.path.exists(args.val_dir_data):
    os.makedirs(args.val_dir_data)

log = get_logger(os.path.join(args.train_dir, 'log'))

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val



class ContiFormer(nn.Module):
    def __init__(self, obs_dim, device, batch_size=10):
        super(ContiFormer, self).__init__()
        args_ode = {
            'use_ode': True, 'actfn': 'tanh', 'layer_type': 'concat', 'zero_init': True,
            'atol': args.atol, 'rtol': args.rtol, 'method': args.method, 'regularize': False,
            'approximate_method': 'bilinear', 'nlinspace': 1, 'linear_type': 'before',
            'interpolate': 'linear', 'itol': 1e-2
        }
        args_ode = AttrDict(args_ode)

        self.encoder = EncoderLayer(16, 64, 6, 4, 4, args=args_ode, dropout=args.dropout).to(device)
        self.lin_in = nn.Linear(obs_dim, 16).to(device)
        self.lin_out = nn.Linear(16, obs_dim).to(device)

        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / 16) for i in range(16)])
        self.batch_size = batch_size

    def temporal_enc(self, time):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def pad_input(self, input, t0, tmax=6 * math.pi):
        input_last = input[:, -1:, :]
        input = torch.cat((input, input_last), dim=1)
        t0 = torch.cat((t0, torch.tensor([tmax]).to(t0.device)), dim=0)
        return input, t0

    def forward(self, samples, orig_ts, **kwargs):
        if kwargs.get('is_train', False):
            bs, ls = samples.shape[0], len(orig_ts)
            # sample_idx = npr.choice(bs, self.batch_size, replace=False)
            # samples = samples[sample_idx, ...]

            t0 = samples[..., -1]
            input = self.lin_in(samples[..., :-1])
            input = (input + self.temporal_enc(t0)).float()

            _input, _t0 = self.pad_input(input, t0[0])

            X = torchcde.LinearInterpolation(_input, t=_t0)
            input = X.evaluate(orig_ts).float()
            orig_ts = torch.tensor(orig_ts).to(input.device)

            mask = torch.zeros(self.batch_size, ls, 1).to(input.device)
            out, _ = self.encoder(input, orig_ts.unsqueeze(0).repeat(self.batch_size, 1).float(),
                                  mask=mask.bool())
            return self.lin_out(out)
        else:
            bs, ls = samples.shape[0], len(orig_ts)
            t0 = samples[..., -1]
            input = self.lin_in(samples[..., :-1])
            input = (input + self.temporal_enc(t0)).float()

            _input, _t0 = self.pad_input(input, t0[0])

            X = torchcde.LinearInterpolation(_input, t=_t0)
            input = X.evaluate(orig_ts).float()
            orig_ts = torch.tensor(orig_ts).to(input.device)

            mask = torch.zeros(bs, ls, 1).to(input.device)
            out, _ = self.encoder(input, orig_ts.unsqueeze(0).repeat(bs, 1).float(), mask=mask.bool())
            return self.lin_out(out), None

    def calculate_loss(self, pred_x, target_x):
        # pred_x, idx = out
        # target_x, _, _ = target
        # if idx is not None:
        #     return ((pred_x - target_x[idx, ...]) ** 2).sum()
        # else:
        return ((pred_x - target_x) ** 2).sum()


def get_ds_timeSeries(function_args):
    train_count = function_args.train_count
    val_count = function_args.val_count
    x_values = np.arange(0, function_args.number_x_values)

    train_ds = TimeSeriesDataset_Interpolation_roundedInput(train_count, x_values, function_args, args.batch_size)
    val_ds = TimeSeriesDataset_Interpolation_roundedInput(val_count, x_values, function_args, args.batch_size)

    train_dataloader = DataLoader(train_ds, batch_size=function_args.batch_size)
    val_dataloader = DataLoader(val_ds, batch_size=1)

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    np.random.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    obs_dim = 1

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')


    #dataloader 
    train_dataloader, val_dataloader = get_ds_timeSeries(args)
    
    model = ContiFormer(obs_dim, device, args.batch_size)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_meter = RunningAverageMeter()

    st = 0

    if args.train_dir is not None:
        ckpt_path = os.path.join(args.train_dir, f'ckpt_{args.model_name}.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            st = checkpoint['itr']
            log.info('Loaded ckpt from {}'.format(ckpt_path))

    for itr in range(st + 1, args.niters + 1):
        # train one iteration
        # batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {itr:02d}")
        # for batch in batch_iterator:
        #     optimizer.zero_grad()
        #     # backward in time to infer q(z_0)

        #     groundTruth = batch["groundTruth"].unsqueeze(-1).to(device)
        #     timeSeries_noisy_original = batch["noisy_TimeSeries"]
        #     mask = batch["mask"]
        #     time_stamps_original = torch.tensor(np.arange(0, args.number_x_values)).to(device)
            
        #     div_term = batch["div_term"].unsqueeze(-1).unsqueeze(-1).to(device)
        #     min_value = batch["min_value"].unsqueeze(-1).unsqueeze(-1).to(device)
        #     noise_std = batch["noise_std"]

        #     row_idx, col_idx = torch.where(mask)
        #     n_true_per_row = mask.sum(dim=1)[0].item()
        #     indices_per_row = col_idx.view(10, n_true_per_row)
        #     timeSeries_noisy = timeSeries_noisy_original.gather(1, indices_per_row)
        #     indices_per_row = indices_per_row.unsqueeze(-1)
        #     timeSeries_noisy = timeSeries_noisy.unsqueeze(-1)
        #     timeSeries_noisy = torch.cat((timeSeries_noisy, indices_per_row), dim=-1).float().to(device)

        #     # for i in range(args.batch_size):
        #     #     y_values_loop = timeSeries_noisy_original[i]
        #     #     x_values = np.arange(len(y_values_loop))
        #     #     mask_loop = mask[i]
        #     #     y_values_loop[mask_loop == False] = np.nan

        #     #     fig, ax = plt.subplots(1,1)
        #     #     ax.plot(x_values, y_values_loop)
        #     #     plt.show()




        #     out = model(timeSeries_noisy, time_stamps_original, is_train=True)
        #     # out, idx = out
        #     out = (out*div_term) + min_value
        #     groundTruth = (groundTruth*div_term) + min_value
        #     loss = model.calculate_loss(out, groundTruth)
        #     loss.backward()
        #     optimizer.step()
        #     loss_meter.update(loss.item())

        #     log.info('Iter: {}, running loss: {:.4f}'.format(itr, loss_meter.avg))

        #     ckpt_path = os.path.join(args.train_dir, f'ckpt_{args.model_name}.pth')
        #     torch.save({
        #         'model': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'itr': itr,
        #     }, ckpt_path)
        #     log.info('Stored ckpt at {}'.format(ckpt_path))

        # test one iteration
        with torch.no_grad():
            for batch in val_dataloader:

                groundTruth = batch["groundTruth"]
                timeSeries_noisy_original = batch["noisy_TimeSeries"]
                mask = batch["mask"]
                time_stamps_original = torch.tensor(np.arange(0, args.number_x_values))
                
                div_term = batch["div_term"]
                min_value = batch["min_value"]
                noise_std = batch["noise_std"]

                # mask_indices = torch.where(mask[0] == True)[0].to(device)
                # timeSeries_noisy = timeSeries_noisy_original[:,mask_indices].unsqueeze(-1)
                # time_stamps = time_stamps_original.detach().clone()[mask_indices]
                # time_stamps = time_stamps.reshape(1,-1,1).repeat(timeSeries_noisy.size(0),1,1)
                # timeSeries_noisy = torch.cat((timeSeries_noisy, time_stamps), dim=-1).float()

                row_idx, col_idx = torch.where(mask)
                n_true_per_row = mask.sum(dim=1)[0].item()
                indices_per_row = col_idx.view(1, n_true_per_row)
                timeSeries_noisy = timeSeries_noisy_original.gather(1, indices_per_row)
                indices_per_row = indices_per_row.unsqueeze(-1)
                timeSeries_noisy = timeSeries_noisy.unsqueeze(-1)
                timeSeries_noisy = torch.cat((timeSeries_noisy, indices_per_row), dim=-1).float()

                pred_x = model(timeSeries_noisy, time_stamps_original)[0]

                pred_x = (pred_x * div_term) + min_value
                groundTruth = (groundTruth * div_term) + min_value

                mae = torch.abs(pred_x - groundTruth.unsqueeze(-1)).sum(dim=-1).mean()
                rmse = torch.sqrt(((pred_x - groundTruth.unsqueeze(-1)) ** 2).sum(dim=-1).mean())
                log.info('Iter: {}, MAE: {:.4f}, RMSE: {:.4f}'.format(itr, mae.item(), rmse.item()))

                orig_traj = groundTruth[0].cpu().numpy()
                samp_traj = timeSeries_noisy[0].cpu().numpy()

                fig, ax = plt.subplots(1,1)
                ax.plot(time_stamps_original, (timeSeries_noisy_original[0]*div_term[0])+min_value[0], label="Noisy Trajectory")
                ax.plot(time_stamps_original, groundTruth[0], label="True Trajectory")
                ax.plot(time_stamps_original, pred_x[0], label="Prediction Trajectory")
                ax.legend()

                save_path = os.path.join(args.val_dir_pictures, f'vis_{itr}.svg')
                plt.savefig(save_path, dpi=500)
                log.info('Saved visualization figure at {}'.format(save_path))

                save_path = os.path.join(args.val_dir_data, f'pred_{itr}.pkl')
                torch.save({
                    'pred': pred_x,
                    'target': groundTruth,
                    'samp': timeSeries_noisy
                }, save_path)

                log.info('Saved predict file at {}'.format(save_path))

                break
