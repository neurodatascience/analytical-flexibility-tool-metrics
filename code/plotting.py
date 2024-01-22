from typing import Mapping

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

def add_legend(fig_or_ax: plt.Figure | plt.Axes, patch_args_by_label: Mapping[str, Mapping], title=None, **kwargs):
    legend_handles = [
        mpatches.Patch(label=label, **patch_args)
        for label, patch_args in patch_args_by_label.items()
    ]
    legend = fig_or_ax.legend(handles=legend_handles, **kwargs)
    legend.set_title(title)
    return legend

def plot_timeseries(data, x, y, log_scale=True, **kwargs):
    if 'legend' not in kwargs:
        kwargs['legend'] = False
    ax = sns.lineplot(data=data, x=x, y=y, **kwargs)
    if log_scale:
        ax.set_yscale('log')
        ax.minorticks_off()
        _, y_max = ax.get_ylim()
        if y_max < 100:
            ax.set_ylim(top=100)
            
    ax.get_yaxis().set_major_formatter(FormatStrFormatter('%.0f'))
    ax.set_ylim(bottom=1)
    ax.set_xlim(left=min(data[x]), right=max(data[x]))
    sns.despine(ax=ax)
    return ax

def plot_bar(data, x, y, log_scale=False, **kwargs):
    if 'legend' not in kwargs:
        kwargs['legend'] = False
    ax = sns.barplot(data=data, x=x, y=y, **kwargs)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f')
    if log_scale:
        ax.set_yscale('log')
        ax.minorticks_off()
    ax.set_yticks([])
    sns.despine(ax=ax, left=True)
    return ax
