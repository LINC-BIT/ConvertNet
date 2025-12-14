from analysis.scaling_law_utils import read_data_points_v2, find_data_points_with_other_variables_constant
from utils.dl.common.env import set_random_seed
set_random_seed(1)

import numpy as np
from scipy.optimize import fmin_l_bfgs_b, least_squares
from torch import nn 
import random
import torch
from utils.common.log import logger
import math
import scipy
import tqdm
import itertools
import os
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
import copy


class EdgeScalingLaw(nn.Module):
    def __init__(self, n_feature_dim, n_blocks) -> None:
        super(EdgeScalingLaw, self).__init__()

        self.p_sv = nn.Parameter(torch.rand(n_feature_dim))
        self.p_tv = nn.Parameter(torch.rand(n_feature_dim))

        self.n_blocks = n_blocks
        
        n_in = n_feature_dim * 4 + 2
        self.network = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.b1 = nn.Linear(128, 4 + 2 * n_blocks)
        self.b2 = nn.Linear(128, 2 + n_blocks)
        
    def nsl_element(self, x, a, b, c):
        eps = 1e-5
        return a + (b * (x + eps) ** c)
        
    def scaling_law(self, 
                    num_retraining_iters,
                    num_used_samples, 
                    num_params_each_block,
                    p1, p2):
        
        res = 0.
        
        res += self.nsl_element((num_retraining_iters + 1), p1[:, 0], p2[:, 1], p2[:, 0])
        
        res += self.nsl_element((num_used_samples + 1), p1[:, 2], p1[:, 3], p2[:, 1])
            
        for bi in range(self.n_blocks):
            res += self.nsl_element(num_params_each_block[:, bi], p1[:, 3 + bi * 2], p1[:, 4 + bi * 2], p2[:, 2 + bi])
        
        return res
    
    def get_p1_p2(self, source_target_dist_distance,
                kb_loss_in_target_dist,
                *feature_stats):
        
        sm, tm, sv, tv = feature_stats
        sv = (self.p_sv.unsqueeze(0).unsqueeze(2) * sv).mean(1)
        tv = (self.p_tv.unsqueeze(0).unsqueeze(2) * tv).mean(1)
        
        x = torch.cat([
            # pruned_blocks_index,
            # num_retraining_iters.unsqueeze(1) + 1.,
            # num_used_samples.unsqueeze(1) + 1.,
            # model_size.unsqueeze(1),
            source_target_dist_distance.unsqueeze(1),
            kb_loss_in_target_dist.unsqueeze(1),
            sm, tm, sv, tv
        ], dim=1).float()
        
        f = self.network(x)
        p1 = self.b1(f)
        p2 = self.b2(f)
        
        return p1, p2
    
    def forward(self, 
                num_params_each_block,
                num_retraining_iters,
                num_used_samples,
                # model_size,
                source_target_dist_distance,
                kb_loss_in_target_dist,
                *feature_stats):
        
        sm, tm, sv, tv = feature_stats
        sv = (self.p_sv.unsqueeze(0).unsqueeze(2) * sv).mean(1)
        tv = (self.p_tv.unsqueeze(0).unsqueeze(2) * tv).mean(1)
        
        x = torch.cat([
            # pruned_blocks_index,
            # num_retraining_iters.unsqueeze(1) + 1.,
            # num_used_samples.unsqueeze(1) + 1.,
            # model_size.unsqueeze(1),
            source_target_dist_distance.unsqueeze(1),
            kb_loss_in_target_dist.unsqueeze(1),
            sm, tm, sv, tv
        ], dim=1).float()
        
        f = self.network(x)
        p1 = self.b1(f)
        p2 = self.b2(f)

        return self.scaling_law(num_retraining_iters,
                                num_used_samples,
                                # model_size,
                                num_params_each_block,
                                p1, p2).float()


def split_train_val_data(data_points, num_data_points_in_a_retraining, num_params_in_blocks):
    assert len(data_points) % num_data_points_in_a_retraining == 0
    
    num_retrainings = len(data_points) // num_data_points_in_a_retraining
    val_data_points_for_draw = data_points[num_data_points_in_a_retraining * (num_retrainings * 4 // 5): ]
    
    random.shuffle(data_points)
    
    train_data_points = data_points[0: num_data_points_in_a_retraining * (num_retrainings * 4 // 5)]
    val_data_points = data_points[num_data_points_in_a_retraining * (num_retrainings * 4 // 5): ]
    
    return process_data_points(train_data_points, num_params_in_blocks), process_data_points(val_data_points, num_params_in_blocks), \
        process_data_points(val_data_points_for_draw, num_params_in_blocks)


def process_data_points(data_points, num_params_in_blocks):
    X, Y = [], []
    for x, y in data_points:
        X += [[
            # (np.array(x['attns_sparsity']) == max(x['attns_sparsity'])).astype(np.float32),
            np.array([p * (1. - s) for p, s in zip(num_params_in_blocks, x['attns_sparsity'])]),
            x['num_retraining_iters'],
            x['num_retraining_iters'] * x['batch_size'],
            # x['small_model_size'],
            x['source_target_dist_distance'],
            x['large_model_loss_in_target_dist'],
            x['features_stats'][0], 
            x['features_stats'][2], 
            x['features_stats'][1],  
            x['features_stats'][3], 
        ]]
        Y += [y]
    return X, Y


class EdgeScalingLawDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    

def train(data_points_files_path, 
          n_feature_dim,
          n_blocks,
          num_params_in_blocks,
          num_data_points_in_a_retraining,
          num_iters,
          val_freq,
          res_save_dir,
          device,
          only_evaluating=None):
    
    edge_scaling_law = EdgeScalingLaw(n_feature_dim, n_blocks).to(device)
    
    data_points = read_data_points_v2(data_points_files_path)
    (train_X, train_Y), (val_X, val_Y), (val_X_for_draw, val_Y_for_draw) = split_train_val_data(
        data_points,
        num_data_points_in_a_retraining,
        num_params_in_blocks
    )
    print(train_X[0])

    if only_evaluating is not None:
        edge_scaling_law = torch.load(only_evaluating).cpu()
        
        val_X_in_retrainings, val_Y_in_retrainings = [], []
        
        retraining_index = 0
        while True:
            if (retraining_index + 1) * num_data_points_in_a_retraining > len(val_X_for_draw):
                break
            val_X_in_retrainings += [val_X_for_draw[retraining_index * num_data_points_in_a_retraining: (retraining_index + 1) * num_data_points_in_a_retraining]]
            val_Y_in_retrainings += [val_Y_for_draw[retraining_index * num_data_points_in_a_retraining: (retraining_index + 1) * num_data_points_in_a_retraining]]
            retraining_index += 1
        
        for ei in range(5):
            val_X_in_9_retrainings, val_Y_in_9_retrainings = [], []
            # for retraining_index in range(9 * ei, 9 * (ei + 1)):
            for _ in range(9):
                retraining_index = random.randint(0, len(val_X_in_retrainings) - 1)
                val_X_in_9_retrainings += [val_X_for_draw[retraining_index * num_data_points_in_a_retraining: (retraining_index + 1) * num_data_points_in_a_retraining]]
                val_Y_in_9_retrainings += [val_Y_for_draw[retraining_index * num_data_points_in_a_retraining: (retraining_index + 1) * num_data_points_in_a_retraining]]
            evaluate_in_a_retraining(edge_scaling_law, val_X_in_9_retrainings, val_Y_in_9_retrainings)
            plt.savefig(os.path.join(res_save_dir, f'only_evaluating_{ei}.png'))
            plt.clf()

    train_dataset = EdgeScalingLawDataset(train_X, train_Y)
    val_dataset = EdgeScalingLawDataset(val_X, val_Y)
    
    from data import build_dataloader
    batch_size = 128
    train_dataloader = build_dataloader(train_dataset, batch_size=batch_size, num_workers=0, infinite=True, shuffle_when_finite=None)
    val_dataloader = build_dataloader(val_dataset, batch_size=batch_size, num_workers=0, infinite=False, shuffle_when_finite=False)
    train_dataloader_for_val = build_dataloader(train_dataset, batch_size=batch_size, num_workers=0, infinite=False, shuffle_when_finite=False)

    optimizer = torch.optim.Adam([
        dict(params=edge_scaling_law.parameters(), lr=1e-5)
    ])
    
    from utils.dl.common.lr_scheduler import get_step_lr_schedule_with_warmup
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                  get_step_lr_schedule_with_warmup(num_iters // 10, int(num_iters * 2 / 5), 0.1))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(num_iters * 2 / 5), gamma=0.1)
    
    pbar = tqdm.tqdm(range(num_iters))
    
    from utils.dl.common.env import create_tbwriter
    tb_writer = create_tbwriter(os.path.join(res_save_dir, 'tb_log'), True)
    
    best_mean_val_abs_error = 1e3
    
    for iter_index in pbar:
        x, y = next(iter(train_dataloader))
        x = [xi.to(device).float() for xi in x]
        y = y.to(device).float()
        
        edge_scaling_law.train()
        pred = edge_scaling_law(*x)
        reg_loss = F.mse_loss(pred, y)
        loss = reg_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter_index)
        
        # report
        train_abs_error = (abs(pred - y)).mean()
        train_rel_error = (abs(pred - y) / y).mean()
        
        pbar.set_description(f'loss: {loss.item():.6f}, rel_error: {train_rel_error.item():.6f}, '
                             f'abs_error: {train_abs_error.item():.6f}')
        
        tb_writer.add_scalars('loss', {'reg': reg_loss.item()}, iter_index)
        tb_writer.add_scalar('train_error/abs', train_abs_error.item(), iter_index)
        tb_writer.add_scalar('train_error/rel', train_rel_error.item(), iter_index)

        if (iter_index + 1) % val_freq == 0:
            edge_scaling_law.eval()
            with torch.no_grad():
                val_abs_errors, val_rel_errors = [], []
                preds, reals = [], []

                for x, y in val_dataloader:
                    x = [xi.to(device) for xi in x]
                    y = y.to(device)
                    pred = edge_scaling_law(*x)
                    
                    val_abs_errors += [abs(pred - y)]
                    val_rel_errors += [abs(pred - y) / y]
                    preds += [pred]
                    reals += [y]

            preds = torch.cat(preds).detach().cpu().numpy()
            reals = torch.cat(reals).detach().cpu().numpy()
            val_abs_errors = torch.cat(val_abs_errors).detach().cpu().numpy()
            val_rel_errors = torch.cat(val_rel_errors).detach().cpu().numpy()
            
            mean_val_rel_error = np.mean(val_rel_errors)
            std_val_rel_error = np.std(val_rel_errors)
            mean_val_abs_error = np.mean(val_abs_errors)
            std_val_abs_error = np.std(val_abs_errors)
            
            tb_writer.add_scalar('val_error/abs_mean', mean_val_abs_error.item(), iter_index)
            tb_writer.add_scalar('val_error/rel_mean', mean_val_rel_error.item(), iter_index)
            tb_writer.add_scalar('val_error/abs_std', std_val_abs_error.item(), iter_index)
            tb_writer.add_scalar('val_error/rel_std', std_val_rel_error.item(), iter_index)
            
            plt.scatter(reals, preds, alpha=0.1, marker='.')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('real')
            plt.ylabel('pred')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            if mean_val_abs_error < best_mean_val_abs_error:
                best_mean_val_abs_error = mean_val_abs_error
                torch.save(edge_scaling_law, os.path.join(res_save_dir, 'best_edge_scaling_law_fcn.pt'))
                plt.savefig(os.path.join(res_save_dir, 'best_bnsl_fcn_pred_vs_real.png'))
                
            tb_writer.add_figure('pred_real_scatter/val', plt.gcf(), iter_index)
            plt.clf()
            
            torch.save(edge_scaling_law, os.path.join(res_save_dir, 'edge_scaling_law_fcn.pt'))
            
            # train
            with torch.no_grad():
                val_abs_errors, val_rel_errors = [], []
                preds, reals = [], []

                for x, y in train_dataloader_for_val:
                    x = [xi.to(device) for xi in x]
                    y = y.to(device)
                    pred = edge_scaling_law(*x)
                    
                    val_abs_errors += [abs(pred - y)]
                    val_rel_errors += [abs(pred - y) / y]
                    preds += [pred]
                    reals += [y]

            preds = torch.cat(preds).detach().cpu().numpy()
            reals = torch.cat(reals).detach().cpu().numpy()
            plt.scatter(reals, preds, alpha=0.1, marker='.')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('real')
            plt.ylabel('pred')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            tb_writer.add_figure('pred_real_scatter/train', plt.gcf(), iter_index)
            plt.clf()
            
            val_X_in_9_retrainings, val_Y_in_9_retrainings = [], []
            for retraining_index in range(9):
                val_X_in_9_retrainings += [val_X_for_draw[retraining_index * num_data_points_in_a_retraining: (retraining_index + 1) * num_data_points_in_a_retraining]]
                val_Y_in_9_retrainings += [val_Y_for_draw[retraining_index * num_data_points_in_a_retraining: (retraining_index + 1) * num_data_points_in_a_retraining]]
            evaluate_in_a_retraining(edge_scaling_law, val_X_in_9_retrainings, val_Y_in_9_retrainings)
            tb_writer.add_figure('iteration_vs_acc', plt.gcf(), iter_index)
            plt.clf()
            
            edge_scaling_law = edge_scaling_law.to(device)
            
            
def evaluate_in_a_retraining(edge_scaling_law, val_X_in_9_retrainings, val_Y_in_9_retrainings):
    edge_scaling_law = edge_scaling_law.cpu()
    
    plt.figure(figsize=(6.4 * 1.5, 4.8))
    plt.rc('font', family='Times New Roman, SimSun')
    
    val_abs_errors = []
    val_rel_errors = []
    val_abs_errors_after_50_iters = []
    val_rel_errors_after_50_iters = []
    
    for idx, (val_X, val_Y) in enumerate(zip(val_X_in_9_retrainings, val_Y_in_9_retrainings)):
        
        val_dataset = EdgeScalingLawDataset(val_X, val_Y)
        from data import build_dataloader
        val_dataloader = build_dataloader(val_dataset, batch_size=len(val_X), num_workers=8, infinite=False, shuffle_when_finite=False)
        x, y = next(iter(val_dataloader))

        pred = edge_scaling_law(*x)
        x = x[1].detach().numpy()
        pred = pred.detach().numpy()
        y = y.detach().numpy()
        
        val_abs_errors += [abs(pred - y)]
        val_rel_errors += [abs(pred - y) / y]
        
        val_abs_errors_after_50_iters += [abs(_pred - _y) for _iter, _pred, _y in zip(x, pred, y) if _iter > 50]
        val_rel_errors_after_50_iters += [abs(_pred - _y) / y for _iter, _pred, _y in zip(x, pred, y) if _iter > 50]
        
        plt.subplot(3, 3, idx + 1)
        plt.title(f'目标域 {idx + 1}')
        plt.plot(x, y, label='真实值', linestyle='-')
        plt.plot(x, pred, label='缩放定律预测值', linestyle='--')
        plt.xlabel('训练迭代数')
        plt.ylabel('精度')
        plt.legend()
    
    plt.tight_layout()
    
    mean_val_rel_error = np.mean(val_rel_errors)
    std_val_rel_error = np.std(val_rel_errors)
    mean_val_abs_error = np.mean(val_abs_errors)
    std_val_abs_error = np.std(val_abs_errors)
    
    mean_val_rel_error_after_50_iters = np.mean(val_rel_errors_after_50_iters)
    std_val_rel_error_after_50_iters = np.std(val_rel_errors_after_50_iters)
    mean_val_abs_error_after_50_iters = np.mean(val_abs_errors_after_50_iters)
    std_val_abs_error_after_50_iters = np.std(val_abs_errors_after_50_iters)
    
    print(f'mean_val_abs_error: {mean_val_abs_error}, std_val_abs_error: {std_val_abs_error}, '
          f'mean_val_rel_error: {mean_val_rel_error}, std_val_rel_error: {std_val_rel_error}')
    
    print(f'mean_val_abs_error_after_50_iters: {mean_val_abs_error_after_50_iters}, std_val_abs_error_after_50_iters: {std_val_abs_error_after_50_iters}, '
          f'mean_val_rel_error_after_50_iters: {mean_val_rel_error_after_50_iters}, std_val_rel_error_after_50_iters: {std_val_rel_error_after_50_iters}')
    
    
if __name__ == '__main__':
    from utils.common.exp import get_res_save_dir
    res_save_dir = get_res_save_dir(__file__, tag=sys.argv[1])
    os.makedirs(res_save_dir)
    
    print(res_save_dir)
    
    import shutil
    shutil.copyfile(__file__, os.path.join(res_save_dir, 'script.py'))
    
    train(
        ['dianzixuebao/scaling_law/vit_b_16/img_cls/results/gen_scaling_law_data_points.py/20240425/999998-111723-formal_donot_consider_sparsity/scaling_law_data_points.pth', 'dianzixuebao/scaling_law/vit_b_16/img_cls/results/gen_scaling_law_data_points.py/20240427/999999-000640-formal_donot_consider_sparsity_2nd_run/scaling_law_data_points.pth', 'dianzixuebao/scaling_law/vit_b_16/img_cls/results/gen_scaling_law_data_points.py/20240427/999997-133825-formal_donot_consider_sparsity_3rd_run/scaling_law_data_points.pth', 'dianzixuebao/scaling_law/vit_b_16/img_cls/results/gen_scaling_law_data_points.py/20240428/999998-110144-6/scaling_law_data_points.pth', 'dianzixuebao/scaling_law/vit_b_16/img_cls/results/gen_scaling_law_data_points.py/20240428/999993-155628-7/scaling_law_data_points.pth', 'dianzixuebao/scaling_law/vit_b_16/img_cls/results/gen_scaling_law_data_points.py/20240428/999995-115552-50/scaling_law_data_points_real_target.pth', 'dianzixuebao/scaling_law/vit_b_16/img_cls/results/gen_scaling_law_data_points.py/20240428/999994-140730-60/scaling_law_data_points_real_target.pth'],
        768,
        12,
        [1193344 / 1024**2] * 12,
        51,
        40000,
        1000,
        res_save_dir,
        'cuda',
        only_evaluating='analysis/scaling_law_trial/results/only_consider_pruned_block_index2.py/20240504/999992-155737-trial/best_edge_scaling_law_fcn.pt'
    )
    
    
    # train(
    #     ['dianzixuebao/scaling_law/vit_b_16/img_cls/results/gen_scaling_law_data_points.py/20240428/999995-115552-50/scaling_law_data_points_real_target.pth', 'dianzixuebao/scaling_law/vit_b_16/img_cls/results/gen_scaling_law_data_points.py/20240428/999994-140730-60/scaling_law_data_points_real_target.pth'],
    #     768,
    #     12,
    #     [1193344 / 1024**2] * 12,
    #     51,
    #     40000,
    #     1000,
    #     res_save_dir,
    #     'cuda',
    #     only_evaluating=None
    # )
    