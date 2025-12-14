from vis.api import draw_accs_under_a_retraining_window
from vis.util import *


lw = 2
ls1 = '--'
ls2 = (0, (1, 1))

baseline_time_per_iter = 1

data = {
    'without training-time scaling': {
        'res_json_path': 'exp/online_retraining/vit_b_16/img_cls/results/vit.py/20251214/999998-205250-[]/res.json',
        'time_per_iter': baseline_time_per_iter,
        'linestyle': dict(color=GREY, lw=lw, ls=ls1)
    },
    
    
    'with training-time scaling': {
        'res_json_path': "exp/online_retraining/vit_b_16/img_cls/results/vit.py/20251214/999997-205652-[(10, ('s', 1)), (20, ('s', 2))]/res.json",
        'time_per_iter': baseline_time_per_iter,
        'linestyle': dict(color=RED, lw=lw, ls=ls1)
    }
}

draw_accs_under_a_retraining_window(data, 50, f'exp/online_retraining/vit_b_16/img_cls/accuracy_comparison.png', y_margin=0.1, y_smooth=0., x_in='iter', draw_legend=16)
