from typing import Mapping

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

def add_legend(fig: plt.Figure, palette: Mapping[str, str], title=None):
    legend_handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in palette.items()
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
    )
    legend.set_title(title)
    return fig

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
