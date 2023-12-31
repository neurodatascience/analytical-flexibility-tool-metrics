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
    COL_SECTION,
    COL_NAME,
    COL_CITATIONS,
    COL_CONTAINER_PULLS,
    COL_CONDA_DOWNLOADS_TIMESERIES,
    COL_PYPI_DOWNLOADS_TIMESERIES,
    COL_PYTHON_DOWNLOADS_TOTAL,
    generate_col_standardized,
)
from metrics import compute_metrics

def plot_citations(df_metrics: pd.DataFrame, ax=None) -> plt.Axes:

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
            data_for_df_citations.append({
                COL_NAME: tool,
                col_n_citations: 1,
                col_citation_date: pd.to_datetime(citation['creation']),
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

    ax = sns.lineplot(
        data=df_citations,
        x=col_citation_date,
        y=col_n_citations_cumulative,
        hue=COL_NAME,
        ax=ax,
    )

    return ax

def generate_figures(
        fpath_tools: Path,
        dpath_figs: Path,
        config_dict: Mapping[str, Tuple[str, Callable]] = None,
        ax_height=2,
        ax_width_unit=1.5,
        fpath_metrics_in: Path = None,
        fpath_metrics_out: Path = None,
        overwrite: bool = False,
    ):

    label_citations = 'citations'
    label_stars = 'repo_stars'
    label_forks = 'repo_forks'
    label_containers = 'container_pulls'
    label_python_timeseries = 'python_downloads_timeseries'
    label_python_total = 'python_downloads_total'

    sns.set_theme(context='paper', style='ticks')
    
    if fpath_metrics_in is not None:
        print(f'Loading metrics from {fpath_metrics_in}')
        df_metrics = pd.read_csv(
            fpath_metrics_in,
        )
        idx_not_na = ~df_metrics[COL_CITATIONS].isna()
        df_metrics.loc[idx_not_na, COL_CITATIONS] = df_metrics.loc[idx_not_na, COL_CITATIONS].apply(json.loads)
    else:
        df_metrics = compute_metrics(
            fpath_tools=fpath_tools,
            config_dict=config_dict,
            fpath_metrics_out=fpath_metrics_out,
            overwrite=overwrite,
        )

    # TODO create figs directory if needed
    dpath_figs.mkdir(exist_ok=True)

    for section, df_metrics_section in df_metrics.groupby(COL_SECTION):

        print(f'Generating figures for section: {section}')
        print(f'\tNumber of tools: {len(df_metrics_section)}')

        # determine figure layout
        # citation count
        df_citations = df_metrics_section.loc[~pd.isna(df_metrics_section[COL_CITATIONS])]
        n_citations = df_citations[COL_CITATIONS].apply(len).sum()
        with_citations = n_citations > 0
        # code repo stars/forks
        # container pulls
        df_container_pulls = df_metrics_section.loc[
            ~pd.isna(df_metrics_section[generate_col_standardized(COL_CONTAINER_PULLS)])
        ]
        n_container_pulls = df_container_pulls[COL_CONTAINER_PULLS].sum()
        with_container_pulls = n_container_pulls > 0
        # Python package total downloads
        df_python_downloads = df_metrics_section.loc[
            (
                ~pd.isna(df_metrics_section[generate_col_standardized(COL_PYTHON_DOWNLOADS_TOTAL)])
                | ~pd.isna(df_metrics_section[COL_PYPI_DOWNLOADS_TIMESERIES])
                # | ~pd.isna(df_metrics_section[COL_CONDA_DOWNLOADS_TIMESERIES])
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
            subplot_mosaic.append([label_citations] * 6)
        # TODO repo metrics: half/half
        if with_container_pulls:
            subplot_mosaic.append([label_containers] * 6)
        if with_python_downloads_total:
            subplot_mosaic.append(
                ([label_python_timeseries] * 4) + ([label_python_total] * 2)
            )
        
        if len(subplot_mosaic) == 0:
            warnings.warn(f'No plot data for section {section}, skipping')
            continue

        fig, axes = plt.subplot_mosaic(
            mosaic=subplot_mosaic,
            figsize=(ax_width_unit * 6, ax_height * len(subplot_mosaic)),
        )
        ax_citations = axes.get(label_citations, None)
        ax_containers = axes.get(label_containers, None)
        ax_python_timeseries = axes.get(label_python_timeseries, None)
        ax_python_total = axes.get(label_python_total, None)

        if ax_citations is not None:
            plot_citations(
                df_citations,
                ax=ax_citations,
            )
        
        if ax_containers is not None:
            sns.barplot(
                data=df_container_pulls.sort_values(generate_col_standardized(COL_CONTAINER_PULLS), ascending=False),
                y=generate_col_standardized(COL_CONTAINER_PULLS),
                x=COL_NAME,
                ax=ax_containers,
            )

        if ax_python_total is not None:
            sns.barplot(
                data=df_python_downloads.sort_values(generate_col_standardized(COL_PYTHON_DOWNLOADS_TOTAL), ascending=False),
                y=generate_col_standardized(COL_PYTHON_DOWNLOADS_TOTAL),
                x=COL_NAME,
                ax=ax_python_total,
            )

        sns.despine()
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
    )
