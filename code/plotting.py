import math
from typing import Mapping

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from matplotlib.legend_handler import HandlerBase

class TextHandler(HandlerBase):
    def __init__(self, text_props=None, **kwargs):
        self.text_props = text_props or {}
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        return [mtext.Text(xdescent, ydescent, orig_handle.get_text(),  **self.text_props)]

def add_legend(fig_or_ax: plt.Figure | plt.Axes, patch_args_by_label: Mapping[str, Mapping], title=None, labels_by_section=None, section_label_padding=80, **kwargs):
    if 'title_fontproperties' not in kwargs:
        kwargs['title_fontproperties'] = {'weight': 'bold'}
    legend_handles = []
    if labels_by_section is None or len(labels_by_section) == 0:
        labels_by_section = {'': list(patch_args_by_label.keys())}
    for section, labels in labels_by_section.items():
        legend_handles.append(mtext.Text(text=section, label=' ' * section_label_padding))
        legend_handles.extend([
            mpatches.Patch(label=label, **patch_args_by_label[label])
            for label in labels
        ])
    if len(labels_by_section) == 1:
        legend_handles = legend_handles[1:] # no section title if only one section
    legend = fig_or_ax.legend(handles=legend_handles, handler_map={mtext.Text: TextHandler(text_props={'weight': 'bold'})}, **kwargs)
    legend.set_title(title)
    return legend

def plot_timeseries(data: pd.DataFrame, x, y, log_scale=True, **kwargs):
    if 'legend' not in kwargs:
        kwargs['legend'] = False
    ax = sns.lineplot(data=data, x=x, y=y, **kwargs)
    if log_scale:
        ax.set_yscale('log')
        ax.minorticks_off()
        greatest_power_of_ten = max(2, math.floor(math.log10(data[y].max())))
        if greatest_power_of_ten < 4:
            yticks = [10 ** i for i in range(greatest_power_of_ten + 1)]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks)
    
    ax.get_yaxis().set_major_formatter(FormatStrFormatter('%.0f'))
    ax.set_ylim(bottom=1)
    ax.set_xlim(left=min(data[x]), right=max(data[x]))
    ax.set_xlabel('')
    ax.set_ylabel('')
    sns.despine(ax=ax)
    return ax

def plot_bar(data: pd.DataFrame, x, y, log_scale=False, y_max_factor=1.2, label_fontsize=None, **kwargs):
    if 'legend' not in kwargs:
        kwargs['legend'] = False
    if 'saturation' not in kwargs:
        kwargs['saturation'] = 1
    ax = sns.barplot(data=data, x=x, y=y, **kwargs)
    y_max = max(data[y])
    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.0f}', fontsize=label_fontsize)
    if log_scale:
        ax.set_yscale('log')
        ax.minorticks_off()
        y_max = 10 ** (math.log10(y_max) * y_max_factor)
    else:
        y_max *= y_max_factor

    ax.set_ylim(bottom=1e-1, top=y_max)
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    sns.despine(ax=ax, left=True)
    return ax
