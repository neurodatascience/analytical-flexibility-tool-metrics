# analytical-flexibility-tool-metrics

Metrics for tools described in analytical flexibility review paper

## Instructions

Clone this repository: 
```bash
git clone https://github.com/neurodatascience/analytical-flexibility-tool-metrics.git
```

Move to the newly created directory:
```
cd analytical-flexibility-tool-metrics
```

One of the Python packages (`condastats`) needs to be installed using `conda` (there is a version on PyPI but the installation did not work). Hence, running the code requires creating a `conda` environment. Please refer to the official instructions on how to install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) for your operating system.

Once `conda` is installed, the next step is to create a new `conda` environment with the appropriate Python version. Here we call the environment `metrics`, but it can be any name. The code runs with Python 3.11.6. It might run with other versions but has not been tested.
```bash
conda create --name metrics python=3.11.6
```

Activate the environment:
```bash
conda activate metrics
```

Then, the `condastats` package can be installed with:
```bash
conda install -c conda-forge condastats
```

Finally, the other dependencies can be installed via `pip`. Assuming we are still in the `analytical-flexibility-tool-metrics` directory, the command is:
```bash
pip install -r requirements.txt
```

The latest versions of the required packages will most likely work. The exception is `pandas`, because some `condastats` functions crash with `pandas` later than 2.0.0. In case something does not work, the exact versions used when developing the code are:
* `condastats` 0.2.1
* `matplotlib` 3.8.2
* `pandas` 1.5.3
* `pypistats` 1.5.0
* `python-dotenv` 1.0.0
* `requests` 2.31.0
* `seaborn` 0.13.0

The main script is `code/generate_figures.py`.
<!-- TODO show usage and example command -->
