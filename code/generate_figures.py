#!/usr/bin/env python
import argparse
import json
import warnings
from pathlib import Path
from typing import Callable, Mapping, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.dates import MonthLocator, YearLocator

from utils import (
    COL_CITATIONS,
    COL_COLOR,
    COL_CONTAINER_PULLS,
    COL_CONDA_DOWNLOADS_TIMESERIES,
    COL_REPO_FORKS,
    COL_REPO_STARS,
    COL_NAME,
    COL_PYPI_DOWNLOADS_TIMESERIES,
    COL_PYTHON_DOWNLOADS_TOTAL,
    COL_SECTION,
    generate_col_standardized,
)
from metrics import compute_metrics
from plotting import add_legend, plot_timeseries, plot_bar

CITATION_CORRECTIONS = {
    # BIDS Apps
    '10.1371/journal.pcbi.1005209': {
        '10.1007/978-1-0716-3195-9_21': '2023-07-23',   # originally 2012-02-24
        '10.1007/978-1-0716-3195-9_8': '2023-07-23',    # originally 2012-02-24
    },
    # Nextflow
    '10.1038/nbt.3820': {
        '10.1051/jbio/2017029': '2018-02-07',       # originally 2017 (but 2018 for online version)
        '10.1017/s0956796817000119': '2017-10-24',  # originally 2017
    },
    # DataLad
    '10.21105/joss.03262': {
        '10.1101/2021.02.10.430678': '2021-08-23',  # originally 2021-02-11
    },
}

def plot_citations(df_metrics: pd.DataFrame, ax=None, date_corrections=None, palette=None) -> plt.Axes:

    if date_corrections is None:
        date_corrections = {}

    # temporary helper columns
    col_citation_date = 'citation_date'
    col_n_citations = 'n_citations'
    col_n_citations_cumulative = 'n_citations_cumulative'

    # subset columns
    df_metrics = df_metrics.loc[:, [COL_NAME, COL_CITATIONS]]

    # unpack citation JSONs and get dates
    data_for_df_citations = []
    for tool, citations in df_metrics.itertuples(index=False):

        # citations is a list of dicts, one for each citation
        for citation in citations:
            try:
                # fix wrong citation dates if needed
                citation_date = date_corrections[citation['cited']][citation['citing']]
            except KeyError:
                citation_date = citation['creation']
            data_for_df_citations.append({
                COL_NAME: tool,
                col_n_citations: 1,
                col_citation_date: pd.to_datetime(citation_date),
            })

        # add one last entry for the current date
        data_for_df_citations.append({
            COL_NAME: tool,
            col_n_citations: 0,
            col_citation_date: pd.to_datetime('today'),
        })

    if len(data_for_df_citations) == 0:
        warnings.warn('No citations found, skipping plot')
    
    df_citations = pd.DataFrame(data_for_df_citations).sort_values(
        [COL_NAME, col_citation_date],
        ignore_index=True,
    )
    df_citations[col_n_citations_cumulative] = df_citations.groupby(COL_NAME)[col_n_citations].cumsum()

    ax = plot_timeseries(
        data=df_citations,
        x=col_citation_date,
        y=col_n_citations_cumulative,
        hue=COL_NAME,
        ax=ax,
        palette=palette,
    )

    ax.set_title('Citations over time')
    ax.xaxis.set_major_locator(YearLocator())

    return ax

def plot_repo(df_metrics: pd.DataFrame, ax=None, hatches=None, palette=None) -> plt.Axes:
    if hatches is None:
        hatches = [None, '/']
    if len(hatches) != 2:
        raise ValueError(
            'hatches must be a list of length 2'
            ' (one hatch style for each of repo stars and forks)'
            f', got {hatches}'
        )
    
    n_tools = len(df_metrics)
    
    col_metric = 'metric'
    col_value = 'value'
    df_repo = pd.melt(
        df_metrics,
        id_vars=[COL_NAME],
        value_vars=[COL_REPO_STARS, COL_REPO_FORKS],
        var_name=col_metric,
        value_name=col_value,
    )
    ax = plot_bar(
        data=df_repo,
        x=COL_NAME,
        y=col_value,
        hue=col_metric,
        ax=ax,
        log_scale=True,
        y_max_factor=1.6,
    )

    # set bar colour and hatch style
    for i_tool, tool in enumerate(ax.get_xticklabels()):
        for i_patch, patch in enumerate(ax.patches[i_tool:2*n_tools:n_tools]):
            patch.set_facecolor(palette[tool.get_text()])
            patch.set_hatch(hatches[i_patch])

    # add legend
    add_legend(
        ax,
        patch_args_by_label={
            f'Number of {metric}': {
                'facecolor': 'grey',
                'edgecolor': 'white',
                'hatch': hatch if hatch is None else hatch * 3,
            }
            for metric, hatch in zip(['stars', 'forks'], hatches)
        },
        frameon=False,
        borderpad=0,
        loc='center right',
        bbox_to_anchor=(1, 1),
        labelspacing=0.25,
    )

    ax.set_title('Code repository metrics')

    return ax

def plot_containers_pulls(df_metrics: pd.DataFrame, ax=None, palette=None) -> plt.Axes:
    ax = plot_bar(
        data=df_metrics,
        x=COL_NAME,
        y=COL_CONTAINER_PULLS,
        ax=ax,
        hue=COL_NAME,
        palette=palette,
        log_scale=True,
    )
    ax.set_title('Container pulls')
    return ax

def plot_python_timeseries(df_metrics: pd.DataFrame, ax=None, palette=None) -> plt.Axes:

    df_python_metrics = df_metrics.loc[
        :,
        [
            COL_NAME,
            COL_PYPI_DOWNLOADS_TIMESERIES,
            COL_CONDA_DOWNLOADS_TIMESERIES,
        ],
    ]

    col_date = 'date'
    col_downloads = 'downloads'
    col_downloads_cumulative = 'downloads_cumulative'

    data_for_df_downloads = []
    min_date_pypi_all_tools = None
    for tool, pypi_download_entries, conda_download_entries in df_python_metrics.itertuples(index=False):

        min_date_pypi = None
        for entry in pypi_download_entries['data']:
            n_downloads = entry['downloads']
            download_date = pd.to_datetime(entry['date'])
            data_for_df_downloads.append({
                COL_NAME: tool,
                col_date: download_date,
                col_downloads: n_downloads,
            })
            if min_date_pypi is None:
                min_date_pypi = download_date
            else:
                min_date_pypi = min(min_date_pypi, download_date)

        if isinstance(conda_download_entries, list):
            for entry in conda_download_entries:
                date = pd.to_datetime(entry['time'])
                if date < min_date_pypi:
                    continue
                data_for_df_downloads.append({
                    COL_NAME: tool,
                    col_date: date,
                    col_downloads: entry['counts'],
                })

        if min_date_pypi_all_tools is None:
            min_date_pypi_all_tools = min_date_pypi
        else:
            min_date_pypi_all_tools = min(min_date_pypi_all_tools, min_date_pypi)

    # add initial entry for all tools
    for tool in df_metrics[COL_NAME]:
        data_for_df_downloads.append({
            COL_NAME: tool,
            col_date: min_date_pypi,
            col_downloads: 0,
        })
    
    df_downloads = pd.DataFrame(data_for_df_downloads).sort_values([COL_NAME, col_date, col_downloads])
    df_downloads[col_downloads_cumulative] = df_downloads.groupby(COL_NAME)[col_downloads].cumsum()

    ax = plot_timeseries(
        data=df_downloads,
        x=col_date,
        y=col_downloads_cumulative,
        hue=COL_NAME,
        ax=ax,
        palette=palette,
    )

    ax.set_title('Python package downloads in the last 180 days')
    ax.xaxis.set_major_locator(MonthLocator())

    return ax

def plot_python_total(df_metrics: pd.DataFrame, ax=None, palette=None) -> plt.Axes:
    ax = plot_bar(
        data=df_metrics,
        y=generate_col_standardized(COL_PYTHON_DOWNLOADS_TOTAL),
        x=COL_NAME,
        ax=ax,
        hue=COL_NAME,
        palette=palette,
    )
    ax.set_title('Total Python package downloads')
    return ax

def process_palette(df_metrics: pd.DataFrame, palette=None) -> Mapping[str, str]:
    if palette is None:
        palette = {}
        for _, df_colors in df_metrics.groupby(COL_SECTION):
            tools = df_colors[COL_NAME]
            colors = sns.color_palette(n_colors=len(df_colors))
            palette.update({
                tool: color
                for tool, color in zip(tools, colors)
            })
    elif COL_COLOR in df_metrics.columns:
        palette = {
            tool: color
            for tool, color
            in df_metrics[[COL_NAME, COL_COLOR]].itertuples(index=False)
        }
    else:
        # validate custom palette
        if not isinstance(palette, Mapping):
            raise TypeError(
                'palette must be a Mapping (tool_name -> color)'
                f', got {type(palette)}'
            )
        for tool in df_metrics[COL_NAME]:
            if tool not in palette:
                raise ValueError(
                    f'custom palette is missing color for tool {tool}'
                )
            
    return palette

def generate_figures(
        fpath_tools: Path,
        dpath_figs: Path,
        config_dict: Mapping[str, Tuple[str, Callable]] = None,
        ax_height=2,
        fig_width=8,
        fpath_metrics_in: Path = None,
        fpath_metrics_out: Path = None,
        overwrite: bool = False,
        citation_corrections: Mapping[str, Mapping[str, str]] = None,
        palette = None,
    ):

    label_citations = 'citations'
    label_repo = 'repo'
    label_containers = 'container_pulls'
    label_python_timeseries = 'python_downloads_timeseries'
    label_python_total = 'python_downloads_total'

    sns.set_theme(style='ticks')
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.titleweight'] = 'bold'
    
    if fpath_metrics_in is not None:
        print(f'Loading metrics from {fpath_metrics_in}')
        df_metrics = pd.read_csv(
            fpath_metrics_in,
        )
        for col in [COL_CITATIONS, COL_PYPI_DOWNLOADS_TIMESERIES, COL_CONDA_DOWNLOADS_TIMESERIES]:
            if col not in df_metrics.columns:
                continue
            idx_not_na = df_metrics[col].notna()
            df_metrics.loc[idx_not_na, col] = df_metrics.loc[idx_not_na, col].apply(json.loads)
    else:
        df_metrics = compute_metrics(
            fpath_tools=fpath_tools,
            config_dict=config_dict,
            fpath_metrics_out=fpath_metrics_out,
            overwrite=overwrite,
        )

    # create figs directory if needed
    dpath_figs.mkdir(exist_ok=True)

    # process palette
    palette = process_palette(df_metrics, palette=palette)


    for section, df_metrics_section in df_metrics.groupby(COL_SECTION):

        print(f'Generating figures for section: {section}')
        print(f'\tNumber of tools: {len(df_metrics_section)}')

        # determine figure layout
        # citation count
        df_citations = df_metrics_section.loc[pd.notna(df_metrics_section[COL_CITATIONS])]
        n_citations = df_citations[COL_CITATIONS].apply(len).sum()
        with_citations = n_citations > 0
        # code repo stars/forks
        df_repo = df_metrics_section.loc[
            pd.notna(
                df_metrics_section[[COL_REPO_STARS, COL_REPO_FORKS]]
            ).any(axis='columns')
        ]
        with_repo = len(df_repo) > 0
        # container pulls
        df_container_pulls = df_metrics_section.loc[
            pd.notna(df_metrics_section[generate_col_standardized(COL_CONTAINER_PULLS)])
        ]
        n_container_pulls = df_container_pulls[COL_CONTAINER_PULLS].sum()
        with_container_pulls = n_container_pulls > 0
        # Python package total downloads
        df_python_downloads = df_metrics_section.loc[
            (
                pd.notna(df_metrics_section[generate_col_standardized(COL_PYTHON_DOWNLOADS_TOTAL)])
                | pd.notna(df_metrics_section[COL_PYPI_DOWNLOADS_TIMESERIES])
                | pd.notna(df_metrics_section[COL_CONDA_DOWNLOADS_TIMESERIES])
            )
        ]
        n_python_downloads_total = df_python_downloads[COL_PYTHON_DOWNLOADS_TOTAL].sum()
        with_python_downloads_total = n_python_downloads_total > 0

        print(f'\tTotal number of citations: {n_citations}')
        print(f'\tTotal number of container pulls: {n_container_pulls}')
        print(f'\tTotal number of Python package downloads: {n_python_downloads_total}')

        # initialize figure with the appropriate layout

        subplot_mosaic = []
        if with_citations:
            subplot_mosaic.append([label_citations])
        if with_repo:
            subplot_mosaic.append([label_repo])
        if with_container_pulls:
            subplot_mosaic.append([label_containers])
        if with_python_downloads_total:
            subplot_mosaic.append([label_python_timeseries])
        
        if len(subplot_mosaic) == 0:
            warnings.warn(f'No plot data for section {section}, skipping')
            continue

        fig, axes = plt.subplot_mosaic(
            mosaic=subplot_mosaic,
            figsize=(fig_width, ax_height * len(subplot_mosaic)),
        )
        ax_citations = axes.get(label_citations, None)
        ax_repo = axes.get(label_repo, None)
        ax_containers = axes.get(label_containers, None)
        ax_python_timeseries = axes.get(label_python_timeseries, None)
        ax_python_total = axes.get(label_python_total, None)

        if ax_citations is not None:
            plot_citations(
                df_citations,
                ax=ax_citations,
                date_corrections=citation_corrections,
                palette=palette,
            )

        if ax_repo is not None:
            plot_repo(
                df_repo,
                ax=ax_repo,
                palette=palette,
            )

        if ax_containers is not None:
            plot_containers_pulls(
                df_container_pulls,
                ax=ax_containers,
                palette=palette,
            )

        if ax_python_timeseries is not None:
            plot_python_timeseries(
                df_metrics_section.loc[df_metrics_section[COL_PYPI_DOWNLOADS_TIMESERIES].notna()],
                ax=ax_python_timeseries,
                palette=palette,
            )

        if ax_python_total is not None:
            plot_python_total(
                df_python_downloads,
                ax=ax_python_total,
                palette=palette,
            )

        add_legend(
            fig,
            patch_args_by_label={
                tool: {
                    'color': palette[tool]
                }
                for tool in df_metrics_section[COL_NAME]
            },
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )

        # add panel labels
        for i_ax, ax in enumerate(axes.values()):
            ax: plt.Axes
            ax.text(
                x=-0.05,
                y=1.08,
                s=chr(ord('A') + i_ax),
                transform=ax.transAxes,
                fontsize=14,
                fontweight='bold',
                va='bottom',
                ha='right',
            )

        fig.tight_layout()

        section_name_clean = section.lower()
        for char in ' .-':
            section_name_clean = section_name_clean.replace(char, '_')
        fpath_fig = dpath_figs / f'{section_name_clean}.png'
        fig.savefig(fpath_fig, dpi=300, bbox_inches='tight')

if __name__ == '__main__':

    dpath_root = Path(__file__).parent.parent
    default_fpath_tools = dpath_root / 'data' / 'tools.csv'
    default_dpath_figs = dpath_root / 'figs'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tools',
        dest='fpath_tools',
        help=f'path to CSV file containing information about tools (default: {default_fpath_tools})',
        type=Path,
        default=default_fpath_tools,
        required=False,
    )
    parser.add_argument(
        '--figs-dir',
        dest='dpath_figs',
        help=f'path to output figures directory (default: {default_dpath_figs})',
        type=Path,
        default=default_dpath_figs,
        required=False,
    )
    parser.add_argument(
        '--load-metrics',
        dest='metrics_csv_in',
        help=(
            'path to read metrics CSV file (optional). '
            'Note: --load-metrics and --save-metrics cannot both be specified. '
            'Also, if --load-metrics is specified, --tools is ignored'
        ),
        type=Path,
        default=None,
        required=False,
    )
    parser.add_argument(
        '--save-metrics',
        dest='metrics_csv_out',
        help='path to write metrics CSV file (optional). Note: --load-metrics and --save-metrics cannot both be specified',
        type=Path,
        default=None,
        required=False,
    )
    parser.add_argument(
        '--overwrite',
        help='overwrite existing figures (and metrics file if applicable)',
        action='store_true',
    )

    args = parser.parse_args()
    fpath_tools = args.fpath_tools
    dpath_figs = args.dpath_figs
    fpath_metrics_in = args.metrics_csv_in
    fpath_metrics_out = args.metrics_csv_out
    overwrite = args.overwrite

    if fpath_tools is None and fpath_metrics_in is None:
        raise RuntimeError(
            'One of --tools and --load-metrics must be specified'
        )
    if fpath_tools is not None and fpath_metrics_in is not None:
        warnings.warn(
            'Both --tools and --load-metrics specified, ignoring --tools'
        )
    if fpath_metrics_in is not None and fpath_metrics_out is not None:
        raise RuntimeError(
            '--load-metrics and --save-metrics cannot both be specified'
        )

    generate_figures(
        fpath_tools=fpath_tools,
        dpath_figs=dpath_figs,
        fpath_metrics_in=fpath_metrics_in,
        fpath_metrics_out=fpath_metrics_out,
        overwrite=overwrite,
        citation_corrections=CITATION_CORRECTIONS,
    )
