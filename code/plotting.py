import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

def plot_timeseries(data, x, y, log_scale=True, **kwargs):
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
    ax = sns.barplot(data=data, x=x, y=y, **kwargs)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f')
    if log_scale:
        ax.set_yscale('log')
        ax.minorticks_off()
    ax.set_yticks([])
    sns.despine(ax=ax, left=True)
    return ax
