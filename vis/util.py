import matplotlib.pyplot as plt
import os


BLUE = (45./255., 164./255., 205./255.)
GREEN = (1./255., 113./255., 0./255.)
YELLOW = (205/255., 194/255., 45/255.)
PURPLE = (204/255., 46/255., 206/255.)
GREY = (146./255., 146./255., 146./255.)
BLACK = (60./255., 60./255., 60./255.)
RED = (181./255., 23./255., 0./255.)


def set_figure_settings(fig_wh_ratio=6.4/4.8, std_h=4.8, font_size=24, font_family='Times New Roman'):
    fig = plt.figure(figsize=(std_h * fig_wh_ratio, std_h))
    if font_family is not None:
        plt.rc('font', family=font_family) # 'Cambria'
    plt.rcParams['font.size'] = str(font_size)

    return fig


def make_alpha(c, alpha):
    return (*c, alpha)