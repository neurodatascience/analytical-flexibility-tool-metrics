COL_NAME = 'tool_name'
COL_SECTION = 'review_paper_section'
COL_DOI = 'doi'
COL_GITHUB = 'github'
COL_GITLAB = 'gitlab'
COL_DOCKER1 = 'docker1'
COL_DOCKER2 = 'docker2'
COL_GITHUB_CONTAINER = 'github_container'
COL_PYPI = 'pypi'
COL_CONDA = 'conda'

COL_CITATIONS = 'citations'
COL_GITHUB_STARS = 'github_stars'
COL_GITHUB_FORKS = 'github_forks'
COL_GITLAB_STARS = 'gitlab_stars'
COL_GITLAB_FORKS = 'gitlab_forks'
COL_DOCKER_PULLS1 = 'docker_pulls1'
COL_DOCKER_PULLS2 = 'docker_pulls2'
COL_GITHUB_CONTAINER_PULLS = 'github_container_pulls'
COL_PYPI_DOWNLOADS_TIMESERIES = 'pypi_downloads_timeseries'
COL_PYPI_DOWNLOADS_TOTAL = 'pypi_downloads_total'
COL_CONDA_DOWNLOADS_TIMESERIES = 'conda_downloads'
COL_CONDA_DOWNLOADS_TOTAL = 'conda_downloads_total'
COLS_TO_STANDARDIZE = [
    COL_GITHUB_STARS,
    COL_GITHUB_FORKS,
    COL_GITLAB_STARS,
    COL_GITLAB_FORKS,
    COL_DOCKER_PULLS1,
    COL_DOCKER_PULLS2,
    COL_GITHUB_CONTAINER_PULLS,
    COL_PYPI_DOWNLOADS_TOTAL,
    COL_CONDA_DOWNLOADS_TOTAL,
]
COLS_STARS = [COL_GITHUB_STARS, COL_GITLAB_STARS]
COLS_FORKS = [COL_GITHUB_FORKS, COL_GITLAB_FORKS]
COLS_CONTAINER_PULLS = [COL_DOCKER_PULLS1, COL_DOCKER_PULLS2, COL_GITHUB_CONTAINER_PULLS]
COLS_PYTHON_DOWNLOADS_TOTAL = [COL_PYPI_DOWNLOADS_TOTAL, COL_CONDA_DOWNLOADS_TOTAL]
COL_CONTAINER_PULLS = 'container_pulls'  # total count
COL_PYTHON_DOWNLOADS_TOTAL = 'python_downloads_total'  # total count
SUFFIX_DATE = '_date'
SUFFIX_STANDARDIZED = '_standardized'

def generate_col_date(col):
    return f'{col}{SUFFIX_DATE}'

def generate_col_standardized(col):
    return f'{col}{SUFFIX_STANDARDIZED}'
