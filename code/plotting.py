import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

def plot_timeseries(data, x, y, **kwargs):
    ax = sns.lineplot(data=data, x=x, y=y, **kwargs)
    ax.set_yscale('log')
    ax.minorticks_off()
    ax.get_yaxis().set_major_formatter(FormatStrFormatter('%.0f'))
    ax.set_ylim(bottom=1)
    ax.set_xlim(left=min(data[x]), right=max(data[x]))
    sns.despine(ax=ax)
    return ax

def plot_bar(data, x, y, **kwargs):
    ax = sns.barplot(data=data, x=x, y=y, **kwargs)
    ax.bar_label(ax.containers[-1], fmt='%.0f')
    ax.set_yticks([])
    sns.despine(ax=ax, left=True)
    return ax
