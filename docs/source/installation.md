# Installation

## Supported environment

HomiCSx 1.0.0 is tested with Python 3.10, DOLFINx 0.9.0, and
`dolfinx_mpc` 0.9.0 on Linux. Windows users should run the Linux
environment under WSL2. macOS may work, but is not currently exercised by
the automated test suite.

The compiled FEniCSx, PETSc, and MPI dependencies are not reliably installed
from PyPI. Consequently, `environment.yml` is the authoritative runtime
dependency specification; the Python package metadata intentionally does not
claim that `pip` can resolve the solver stack.

## Install from source

### 1. Clone the repository

```bash
git clone https://github.com/AmirTahouni/HomiCSx.git
cd HomiCSx
```
### 2. Create the runtime environment

```bash
conda env create -f environment.yml
conda activate homicsx_env
```

### 3. Install HomiCSx

```bash
python -m pip install --no-deps -e .
```

`--no-deps` is deliberate: Conda has already installed the versioned solver
stack. Omit `-e` for a non-editable installation.

### 4. Verify the installation

```bash
python -c "import homicsx; print(homicsx.__version__)"
```

For development, create `environment-dev.yml` instead; it contains the same
solver stack plus the test, coverage, and documentation tools. You can then run
`python -m pytest -q` and build the documentation locally.

## Other dependency versions

Other Python or solver versions may work, but are outside the currently tested
support envelope. If you test another configuration successfully, a concise
compatibility report or pull request is welcome.

