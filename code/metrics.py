import json
import os
import requests
import warnings
from datetime import datetime
from functools import wraps
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple

import pandas as pd
import pypistats
from condastats.cli import overall as condastats_overall
from dotenv import load_dotenv

from utils import (
    COL_DOI,
    COL_GITHUB,
    COL_GITLAB,
    COL_DOCKER1,
    COL_DOCKER2,
    COL_GITHUB_CONTAINER,
    COL_PYPI,
    COL_CONDA,
    COL_CITATIONS,
    COL_GITHUB_STARS,
    COL_GITHUB_FORKS,
    COL_GITLAB_STARS,
    COL_GITLAB_FORKS,
    COL_DOCKER_PULLS1,
    COL_DOCKER_PULLS2,
    COL_GITHUB_CONTAINER_PULLS,
    COL_PYPI_DOWNLOADS_TIMESERIES,
    COL_PYPI_DOWNLOADS_TOTAL,
    COL_CONDA_DOWNLOADS_TIMESERIES,
    COL_CONDA_DOWNLOADS_TOTAL,
    COL_CONTAINER_PULLS,
    COL_PYTHON_DOWNLOADS_TOTAL,
    COLS_TO_STANDARDIZE,
    COLS_CONTAINER_PULLS,
    COLS_PYTHON_DOWNLOADS_TOTAL,
    generate_col_date,
    generate_col_standardized,
)

_REQUESTS_CACHE = {}

_CONDASTATS_CACHE = {}

def get_response_json(url, **kwargs):
    if url in _REQUESTS_CACHE:
        response = _REQUESTS_CACHE[url]
    else:
        try:
            response = requests.get(url, **kwargs)
            if not response.ok:
                print(json.dumps(response.json(), indent=4))
                raise RuntimeError
        except Exception:
            raise RuntimeError(
                f'{response.status_code} error for HTTP GET request at {url}'
            )

        _REQUESTS_CACHE[url] = response
    
    return response.json()

def get_condastats_overall(conda_package):
    if conda_package not in _CONDASTATS_CACHE:
        _CONDASTATS_CACHE[conda_package] = condastats_overall(conda_package, complete=True)
    return _CONDASTATS_CACHE[conda_package]

def get_github_response(github_repo):
    
    try:
        username = os.getenv('GITHUB_USERNAME')
        token = os.getenv('GITHUB_TOKEN')
        auth = (username, token)
    except KeyError:
        auth=None
    url = f'https://api.github.com/repos/{github_repo}'
    return get_response_json(url, auth=auth)

def get_gitlab_response(project_id):
    url = f'https://gitlab.com/api/v4/projects/{project_id}'
    return get_response_json(url)

def get_docker_response(dockerhub_id):
    url = f'https://hub.docker.com/v2/repositories/{dockerhub_id}'
    return get_response_json(url)

def metric(func):
    @wraps(func)
    def _metric(source):
        if pd.isna(source):
            return pd.NA
        return func(source)
    return _metric

@metric
def get_citations(doi):
    url = f'https://opencitations.net/index/coci/api/v1/citations/{doi}'
    return get_response_json(url)

@metric
def get_github_stars(github_repo):
    return get_github_response(github_repo)['stargazers_count']

@metric
def get_github_forks(github_repo):
    return get_github_response(github_repo)['forks_count']

@metric
def get_github_date(github_repo):
    return pd.to_datetime(get_github_response(github_repo)['created_at'])

@metric
def get_gitlab_stars(project_id):
    return get_gitlab_response(project_id)['star_count']

@metric
def get_gitlab_forks(project_id):
    return get_gitlab_response(project_id)['forks_count']

@metric
def get_gitlab_date(project_id):
    return pd.to_datetime(get_gitlab_response(project_id)['created_at'])

@metric
def get_dockerhub_pulls(dockerhub_id):
    return get_docker_response(dockerhub_id)['pull_count']

@metric
def get_dockerhub_date(dockerhub_id):
    return pd.to_datetime(get_docker_response(dockerhub_id)['date_registered'])

@metric
def get_github_container_pulls(github_container_id):
    token_url = f'https://ghcr.io/token?scope=repository:{github_container_id}:pull'
    token = get_response_json(token_url)['token']
    manifest_url = f'https://ghcr.io/v2/{github_container_id}/manifests/latest'
    # works but doesn't have pulls information
    print(
        json.dumps(
            get_response_json(
                manifest_url,
                headers={
                    'Authorization': f'Bearer {token}',
                }
            ),
            indent=4,
        )
    )
    # return None

@metric
def get_pypi_downloads_recent(pypi_package):
    return json.loads(pypistats.overall(pypi_package, format='json', mirrors=False, total='daily'))

@metric
def get_pypi_downloads_total(pypi_package):
    data = json.loads(pypistats.overall(pypi_package, format='json', mirrors=False))['data']
    if len(data) != 1:
        raise RuntimeError(
            'Expected pypistats call to only have 1 entry for package'
            f' {pypi_package}, but got {len(data)}'
        )
    return data[0]['downloads']

@metric
def get_pypi_date(pypi_package):
    url = f'https://pypi.org/pypi/{pypi_package}/json'
    releases = get_response_json(url)['releases']
    dates = []
    for release_tags in releases:
        dates.extend([
            pd.to_datetime(entry['upload_time'])
            for entry in releases[release_tags]
        ])
    return min(dates)

@metric
def get_conda_downloads(conda_package):

    downloads_info: pd.DataFrame = get_condastats_overall(conda_package)

    if len(downloads_info) == 0:
        return pd.NA
    
    return downloads_info.to_json(orient='records')

@metric
def get_conda_downloads_total(conda_package):

    downloads_info: pd.DataFrame = get_condastats_overall(conda_package)

    if len(downloads_info) == 0:
        return pd.NA
    
    return downloads_info['counts'].sum()

@metric
def get_conda_date(conda_package):

    downloads_info: pd.DataFrame = get_condastats_overall(conda_package)

    if len(downloads_info) == 0:
        return pd.NA
    
    return pd.to_datetime(downloads_info['date']).min()

def compute_metrics(
        fpath_tools: Path,
        config_dict: Optional[Mapping[str, Tuple[str, Callable[[str], Any]]]] = None,
        fpath_metrics_out: Optional[Path] = None,
        fpath_dotenv: Optional[Path] = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:

    def combine_cols(df, col_combined, cols_to_combine, combine_standardized=True):

        def add_col(df, col_source, col_target):
            if col_target not in df.columns:
                df[col_target] = pd.NA
            if col_source in df.columns:
                df[col_target] = df[col_target].add(df[col_source], fill_value=0)
            return df

        for col_to_combine in cols_to_combine:
            df = add_col(df, col_to_combine, col_combined)
            if combine_standardized:
                df = add_col(
                    df,
                    generate_col_standardized(col_to_combine),
                    generate_col_standardized(col_combined),
                )
        return df

    cols_json = [COL_CITATIONS, COL_PYPI_DOWNLOADS_TIMESERIES, COL_CONDA_DOWNLOADS_TIMESERIES]

    if fpath_metrics_out is not None and fpath_metrics_out.exists() and not overwrite:
        raise RuntimeError(
            f'{fpath_metrics_out} already exists. Use --overwrite to overwrite'
        )

    if config_dict is None:
        config_dict = {
            COL_CITATIONS: (COL_DOI, get_citations),
            COL_GITHUB_STARS: (COL_GITHUB, get_github_stars),
            COL_GITHUB_FORKS: (COL_GITHUB, get_github_forks),
            generate_col_date(COL_GITHUB_STARS): (COL_GITHUB, get_github_date),
            generate_col_date(COL_GITHUB_FORKS): (COL_GITHUB, get_github_date),
            COL_GITLAB_STARS: (COL_GITLAB, get_gitlab_stars),
            COL_GITLAB_FORKS: (COL_GITLAB, get_gitlab_forks),
            generate_col_date(COL_GITLAB_STARS): (COL_GITLAB, get_gitlab_date),
            generate_col_date(COL_GITLAB_FORKS): (COL_GITLAB, get_gitlab_date),
            COL_DOCKER_PULLS1: (COL_DOCKER1, get_dockerhub_pulls),
            generate_col_date(COL_DOCKER_PULLS1): (COL_DOCKER1, get_dockerhub_date),
            COL_DOCKER_PULLS2: (COL_DOCKER2, get_dockerhub_pulls),
            generate_col_date(COL_DOCKER_PULLS2): (COL_DOCKER2, get_dockerhub_date),
            # TODO look into GitHub container registry
            # COL_GITHUB_CONTAINER_PULLS: (COL_GITHUB_CONTAINER, get_github_container_pulls),
            COL_PYPI_DOWNLOADS_TIMESERIES: (COL_PYPI, get_pypi_downloads_recent),
            COL_PYPI_DOWNLOADS_TOTAL: (COL_PYPI, get_pypi_downloads_total),
            generate_col_date(COL_PYPI_DOWNLOADS_TOTAL): (COL_PYPI, get_pypi_date),
            COL_CONDA_DOWNLOADS_TIMESERIES: (COL_CONDA, get_conda_downloads),
            COL_CONDA_DOWNLOADS_TOTAL: (COL_CONDA, get_conda_downloads_total),
            generate_col_date(COL_CONDA_DOWNLOADS_TOTAL): (COL_CONDA, get_conda_date),
        }

    # load 
    if fpath_dotenv is None:
        fpath_dotenv = Path(__file__).parent / '.env'
    if fpath_dotenv.exists():
        print(f'Loading environment variables from {fpath_dotenv}')
        load_dotenv()
    else:
        warnings.warn(
            f'Did not find dotenv file {fpath_dotenv}. '
            'Metrics from APIs that have lower rate limits for '
            'non-authenticated requests (e.g. GitHub) may not be available.'
        )

    df_tools = pd.read_csv(fpath_tools)
    print(f'Generating metrics for {len(df_tools)} tools')

    for col_metric, (col_info, metric_func) in config_dict.items():
        if col_info in df_tools.columns:
            df_tools[col_metric] = df_tools[col_info].apply(metric_func)
        else:
            warnings.warn(
                f'Skipping {col_metric} metric since {col_info}'
                ' is not in the input file'
            )

    # standardize some metrics by date
    fetch_date = pd.to_datetime(datetime.now())  # arbitrary timezone
    for col_to_standardize in COLS_TO_STANDARDIZE:

        col_date = generate_col_date(col_to_standardize)
        col_standardized = generate_col_standardized(col_to_standardize)

        if not col_to_standardize in df_tools.columns:
            continue

        if not col_date in df_tools.columns:
            raise RuntimeError(
                f'Cannot standardize {col_to_standardize}: missing date info'
            )

        idx_not_na = df_tools[col_to_standardize].notna()
        # get time difference in months
        time_diff = (fetch_date.to_period('M') - pd.to_datetime(df_tools.loc[idx_not_na, col_date]).dt.to_period('M')).apply(attrgetter('n'))
        df_tools.loc[idx_not_na, col_standardized] = df_tools.loc[idx_not_na, col_to_standardize] / time_diff

    # TODO combine code repo metrics
    

    # combine container metrics
    df_tools = combine_cols(df_tools, COL_CONTAINER_PULLS, COLS_CONTAINER_PULLS)
    df_tools = combine_cols(df_tools, COL_PYTHON_DOWNLOADS_TOTAL, COLS_PYTHON_DOWNLOADS_TOTAL)

    if fpath_metrics_out is not None:
        df_tools_to_save = df_tools.copy()
        # special handling of quotation marks for columns with JSON data
        for col in cols_json:
            idx_not_na = df_tools_to_save[col].notna()
            df_tools_to_save.loc[idx_not_na, col] = df_tools_to_save.loc[idx_not_na, col].apply(json.dumps)
        df_tools_to_save.to_csv(fpath_metrics_out, index=False)
        print(f'Wrote metrics to {fpath_metrics_out}')

    return df_tools

