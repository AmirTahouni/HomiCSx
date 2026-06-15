# Installation
## Quick Start

The recommended installation uses conda to handle FEniCSx and its MPI dependencies. A pre-configured `environment.yml` is provided in the repository.

### 1. Clone the Repository

```bash
git clone https://github.com/AmirTahouni/HomiCSx.git
cd homicsx
```
### 2. Create the Environment

The prerequisites are listed in the `environment.yml` file. Some or available on conda and some on pypi. Create a dedicated environment and install the prerequisites based on the `environment.yml` file:

```bash
conda create -n homicsx_env python=3.10
conda activate homicsx_env
conda env create -f environment.yml
```

### 3. Pip Install
If you already have a working FEniCSx environment with dolfinx, gmsh, and other dependencies, install directly:

```bash
git clone https://github.com/AmirTahouni/HomiCSx.git
cd homicsx
pip install homicsx
```

