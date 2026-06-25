# Installation

Since HomiCSx uses fenics-dolfinx and dolfinx_mpc, it is currently only available for linux and macos. For windows, it is recommended to use wsl.

## Quick Start
The recommended installation uses conda to handle FEniCSx and its MPI dependencies. For more information visit the [documnetation](https://docs.conda.io/en/latest/).
<!-- [https://docs.conda.io/en/latest/](https://docs.conda.io/en/latest/) -->

### 1. Clone the Repository

```bash
git clone https://github.com/AmirTahouni/HomiCSx.git
cd homicsx
```
### 2. Create the Environment

The prerequisites are listed in the `environment.yml` file. Some are available on conda and some on pypi. Create a dedicated environment and install the prerequisites based on the `environment.yml` file:

```bash
conda create -n homicsx_env python=3.10
conda activate homicsx_env
conda env create -f environment.yml
```

By doing so, a ready-to-use environment with all of the prerequisites installed named `homicsx_env` will be created.

### 3. Pip Install
If you already have a working FEniCSx environment with dolfinx, gmsh, and other dependencies, install directly from the root of the directory:

```bash
pip install homicsx
```

