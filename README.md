# Prototype-based federated learning: reproducible experiments

Source code and measured results for *Robust Aggregation Scores Can Rank a Free-Rider
First: Limits of Verification in Prototype-Based Federated Learning*, by Ali Mohammadi
Ruzbahani, Abbas Yazdinejad and Hadis Karimipour. The code, the results, the prototype
trajectories and the model checkpoints are archived at https://doi.org/10.5281/zenodo.23787745.

| Directory | Contents |
|---|---|
| `passportfl/` | Federation, identity protocol, payload auditor, attacks and aggregation rules |
| `experiments/` | Training, replay, attack and measurement scripts |
| `analysis/` | Result-table export, figure generation and numerical checks |
| `results/` | Measured results as JSON; the trajectories under `results/archives*/` and `results/paired_honest/` are in the data asset |
| `checkpoints/` | Passport experiment models, distributed in the data asset |
| `tables/` | Result panels as CSV and a combined JSON file |
| `figures/` | Result figures as vector PDF and high-resolution PNG |
| `notebooks/` | Reproduction notebook and the catalogue of all experiment commands |
| `tests/` | Protocol, measurement and reproduction checks |

## Installation

Use a Python 3.12 or 3.13 virtual environment. The environment used for the released
checks is recorded in `results/environment.json`. Install a PyTorch build for your
platform first; the CUDA 12.8 wheels used here are

```sh
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## Data asset

The prototype trajectories and the passport experiment checkpoints are in
`Robust-Aggregation-data.zip` in the Zenodo record (CC BY 4.0). They are needed only for
replay mode:

    checkpoints/*.pt                 11 files
    results/archives/*.npz           57 files
    results/archives_rev2/*.npz      36 files
    results/paired_honest/*.npz      10 files

Unpack the archive next to the repository and move the four directories into the
repository root:

```sh
unzip Robust-Aggregation-data.zip
mv Robust-Aggregation-data/checkpoints checkpoints
mv Robust-Aggregation-data/results/archives results/archives
mv Robust-Aggregation-data/results/archives_rev2 results/archives_rev2
mv Robust-Aggregation-data/results/paired_honest results/paired_honest
```

`MANIFEST.json` in the archive lists the size and SHA-256 digest of every file. Fresh
training downloads CIFAR-10 and CIFAR-100 when they are absent and does not need the
archive.

## Run the notebook

Open `notebooks/reproduce.ipynb` with the kernel of this environment (`requirements.txt`
installs Jupyter) and run it from top to bottom.

| Mode | Computation and inputs |
|---|---|
| `archived` (default) | Export all tables, redraw all result figures and check the recorded values from the released JSON; no training and no data asset needed |
| `replay` | Recompute the detector, geometry and attack analyses from the released trajectories and checkpoints; requires the data asset |
| `fresh` | Train new federations and run every downstream analysis; substantial GPU computation |

The notebook imports the repository code and runs every experiment through the job
catalogue in `notebooks/reproduction_workflow.py`. It copies the code and the released
results into a new `.reproduction/run_*` directory and writes all outputs there, so the
released results are never overwritten. `SELECT_GROUPS` limits expensive recomputation to
chosen experiment groups. Each run records commands, runtime, logs and output hashes in
`execution_manifest.json`.

Outputs are eleven result-table panels as CSV, a combined `tables.json`, and the result
figures as PDF and PNG. The numerical matrix of the adaptive figure is also written as
JSON. Figures use Times New Roman when it is installed and otherwise DejaVu Serif; the
choice is recorded.

Fresh training is a statistical replication of the released experiments, not a bitwise
one: hardware, library versions and operating-system entropy affect floating-point
results and passport draws. The research identities derived from seed and client index
must not be used as deployment signing secrets.

## Standalone analysis and checks

The following commands work from the repository root without the notebook:

```sh
python analysis/export_tables.py --check --check-figures
python analysis/verify_results.py
python -m pytest tests -q
python analysis/make_figures.py --output-dir redrawn
python analysis/export_tables.py --out redrawn
```

Table validation compares each released row and column with the values derived from the
result JSON. The last two commands redraw the figures and re-export the tables into
`redrawn/` without touching the released files. The numerical checks in `verify_results.py` apply to the released
measurements; fresh runs may legitimately differ.

## Individual experiments

The complete list of commands, arguments and output files, grouped by experiment, is the
job catalogue in `notebooks/reproduction_workflow.py`.

## License

The code and the measured results in this repository are released under the MIT License
(see `LICENSE`). The trajectories and checkpoints are released under CC BY 4.0.
