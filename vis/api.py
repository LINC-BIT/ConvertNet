import numpy as np
from utils.common.data_record import read_json
import matplotlib.pyplot as plt
from vis.util import *
from utils.dl.common.env import set_random_seed
from utils.common.data import smoothing

set_random_seed(1)

import random



def draw_accs_under_a_retraining_window(methods_data, retraining_window, fig_save_path, y_margin=0.1, y_smooth=0.6, x_in='hour', draw_legend=False):
    # fig = set_figure_settings(2.5, font_size=24, font_family='Times New Roman, SimSun')
    fig_wh_ratio = 2.5
    std_h = 4.8
    font_size = 24
    # plt.rc('font', family=None)
    plt.rcParams['font.size'] = str(font_size)
    fig, ax = plt.subplots(figsize=(std_h * fig_wh_ratio, std_h))
    
    baselines_avg_acc = 0
    ours_acc = 0
    
    retrained_baselines_avg_acc = 0
    non_retrained_baselines_avg_acc = 0
    
    y_min = 1.0
    y_max = 0
    method_i = 0
    
    for method, method_data in methods_data.items():
        res_json_path = method_data['res_json_path']
        time_per_iter = method_data['time_per_iter']
        
        data = read_json(res_json_path)
        X, Y = [], []

        global_X_offset = 0
        x_ticks = [0]
        num_windows = 0
        
        for dist_i, per_dist_data in enumerate(data):
            accs = per_dist_data['accs']
            
            cur_X = [(a['iter'] * time_per_iter) + global_X_offset for a in accs if a['iter'] * time_per_iter <= retraining_window]
            cur_Y = [a['acc'] for a in accs if a['iter'] * time_per_iter <= retraining_window]
            
            cur_X += [global_X_offset + retraining_window]
            cur_Y += [cur_Y[-1]]
            
            if x_in == 'hour':
                cur_X = [x / 3600. for x in cur_X]
            elif x_in == 'iter':
                cur_X = [x for x in cur_X]
            
            X += cur_X
            Y += cur_Y
            
            global_X_offset += retraining_window
            x_ticks += [cur_X[-1]]
            num_windows += 1
        
        avg_acc = sum(Y) / len(Y)
        if 'Ours' not in method:
            baselines_avg_acc += avg_acc
            
            if 'FLAP' not in method and 'SliceGPT' not in method:
                retrained_baselines_avg_acc += avg_acc
            else:
                non_retrained_baselines_avg_acc += avg_acc
                
        else:
            ours_acc = avg_acc
        # print(f'{method} avg acc: {avg_acc:.4f}')
        
        Y = smoothing(Y, y_smooth)
        y_min = min(np.min(Y), y_min)
        y_max = max(np.max(Y), y_max)
        
        ax.plot(X, Y, label=f'{method}', zorder=method_i, **method_data['linestyle'])
        method_i += 1
        
    y_min = np.floor(y_min / y_margin) * y_margin
    y_max = np.ceil(y_max / y_margin) * y_margin
    if x_in == 'hour':
        ax.set_xlabel('时间（小时）')
    elif x_in == 'iter':
        ax.set_xlabel('Retraining Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_yticks(np.linspace(y_min, y_max, num=4), np.linspace(y_min, y_max, num=4))
    try:
        ax.set_xticks(x_ticks, [f'{x:.1f}' if xi % (num_windows // 5) == 0 else '' for xi, x in enumerate(x_ticks)])
    except:
        pass
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    ax.set_ylim(y_min, y_max)
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    # plt.xticks([])
    ax.grid()
    # plt.legend(loc=2, bbox_to_anchor=(1.05, 1.1), fontsize=20, frameon=False)
    
    if draw_legend is not None:
        # plt.legend(loc=2, bbox_to_anchor=(0.45, 1.1), frameon=False, fontsize=draw_legend)
        plt.legend(fontsize=draw_legend)
    
    plt.tight_layout()
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width, box.height * 0.75])
    
    # ax.legend(loc=9, bbox_to_anchor=(0.45, 1.55), fontsize=22, ncol=3, frameon=False, framealpha=1.0) 
    
    
    
    plt.savefig(fig_save_path, dpi=300)
    plt.savefig(fig_save_path.replace('.png', '.svg'), dpi=300)
    
    plt.clf()
    
    print('Figure saved in:')
    print(fig_save_path)
    
