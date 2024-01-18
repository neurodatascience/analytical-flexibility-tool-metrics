#!/usr/bin/env python
import argparse
import json
import warnings
from pathlib import Path
from typing import Callable, Mapping, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import (
    COL_CITATIONS,
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
from plotting import plot_timeseries, plot_bar

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

def plot_citations(df_metrics: pd.DataFrame, ax=None, date_corrections=None) -> plt.Axes:

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
    )

    return ax

def plot_repo(df_metrics: pd.DataFrame, ax=None) -> plt.Axes:
    col_metric = 'metric'
    col_value = 'value'
    df_metrics = df_metrics.sort_values(COL_REPO_STARS, ascending=False)
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
    )
    return ax

def plot_containers_pulls(df_metrics: pd.DataFrame, ax=None) -> plt.Axes:
    ax = plot_bar(
        data=df_metrics.sort_values(generate_col_standardized(COL_CONTAINER_PULLS), ascending=False),
        x=COL_NAME,
        y=generate_col_standardized(COL_CONTAINER_PULLS),
        ax=ax,
    )
    return ax

def plot_python_timeseries(df_metrics: pd.DataFrame, ax=None) -> plt.Axes:

    df_pypi_metrics = df_metrics.loc[
        :,
        [
            COL_NAME,
            COL_PYPI_DOWNLOADS_TIMESERIES,
            COL_PYTHON_DOWNLOADS_TOTAL,
        ],
    ]

    col_date = 'date'
    col_downloads = 'downloads'
    col_downloads_cumulative = 'downloads_cumulative'

    data_for_df_downloads = []
    for tool, download_entries, total_downloads in df_pypi_metrics.itertuples(index=False):

        n_recent_downloads = 0
        min_date = None
        for entry in download_entries['data']:
            n_downloads = entry['downloads']
            download_date = pd.to_datetime(entry['date'])
            data_for_df_downloads.append({
                COL_NAME: tool,
                col_date: download_date,
                col_downloads: n_downloads,
            })
            if min_date is None:
                min_date = download_date
            else:
                min_date = min(min_date, download_date)
            n_recent_downloads += n_downloads

        # add an initial entry
        data_for_df_downloads.append({
            COL_NAME: tool,
            col_date: min_date,
            col_downloads: total_downloads - n_recent_downloads,
        })
    
    # TODO also do conda downloads

    df_downloads = pd.DataFrame(data_for_df_downloads).sort_values([COL_NAME, col_date])
    df_downloads[col_downloads_cumulative] = df_downloads.groupby(COL_NAME)[col_downloads].cumsum()

    ax = plot_timeseries(
        data=df_downloads,
        x=col_date,
        y=col_downloads_cumulative,
        hue=COL_NAME,
        ax=ax,
    )

    return ax

def plot_python_total(df_metrics: pd.DataFrame, ax=None) -> plt.Axes:
    ax = plot_bar(
        data=df_metrics.sort_values(generate_col_standardized(COL_PYTHON_DOWNLOADS_TOTAL), ascending=False),
        y=generate_col_standardized(COL_PYTHON_DOWNLOADS_TOTAL),
        x=COL_NAME,
        ax=ax,
    )
    return ax

def generate_figures(
        fpath_tools: Path,
        dpath_figs: Path,
        config_dict: Mapping[str, Tuple[str, Callable]] = None,
        ax_height=2,
        fig_width=10,
        fpath_metrics_in: Path = None,
        fpath_metrics_out: Path = None,
        overwrite: bool = False,
        citation_corrections: Mapping[str, Mapping[str, str]] = None,
    ):

    label_citations = 'citations'
    label_repo = 'repo'
    label_containers = 'container_pulls'
    label_python_timeseries = 'python_downloads_timeseries'
    label_python_total = 'python_downloads_total'

    sns.set_theme(context='paper', style='ticks')
    
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
                # | pd.notna(df_metrics_section[COL_CONDA_DOWNLOADS_TIMESERIES])
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
            )

        if ax_repo is not None:
            plot_repo(
                df_repo,
                ax=ax_repo,
            )

        
        if ax_containers is not None:
            plot_containers_pulls(
                df_container_pulls,
                ax=ax_containers,
            )

        if ax_python_timeseries is not None:
            plot_python_timeseries(
                df_metrics_section.loc[df_metrics_section[COL_PYPI_DOWNLOADS_TIMESERIES].notna()],
                ax=ax_python_timeseries,
            )

        if ax_python_total is not None:
            plot_python_total(
                df_python_downloads,
                ax=ax_python_total,
            )

        fig.tight_layout()

        section_name_clean = section.lower()
        for char in ' .-':
            section_name_clean = section_name_clean.replace(char, '_')
        fpath_fig = dpath_figs / f'{section_name_clean}.png'
        fig.savefig(fpath_fig, bbox_inches='tight')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'tools_csv',
        help='path to CSV file containing information about tools',
        type=Path,
    )
    parser.add_argument(
        'figs_dir',
        help='path to output figures directory',
        type=Path,
    )
    parser.add_argument(
        '--load-metrics',
        dest='metrics_csv_in',
        help='path to read metrics CSV file (optional). Note: --load-metrics and --save-metrics cannot both be specified',
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
    fpath_tools = args.tools_csv
    dpath_figs = args.figs_dir
    fpath_metrics_in = args.metrics_csv_in
    fpath_metrics_out = args.metrics_csv_out
    overwrite = args.overwrite

    if fpath_metrics_in is None and fpath_metrics_out is None:
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
