# Gaussian perturbation interpretability: code for the paper

This is the code behind every figure in Clarke, Yip & Nikolaou, *Gaussian perturbation interpretability for regression
techniques to understand retrieval black-boxes* (RASTI 2026). It is meant as supplementary material to the paper, and docstrings
cite the paper's sections, equations, figures and tables.

## Repo Overview

```
_CONFIG.py          machine paths (TauREx, cross-sections, CIA) and the settings shared between experiments
download_data.sh    fetches the opacity data into DATA/XSEC and DATA/CIA
run_all.sh          every step of every experiment, then every figure
DATA/               the two observations and their TauREx-III .par files (Tables B1-B2, G1-G2); XSEC/ and CIA/
                    hold the downloaded opacities
src/                the method and the shared code (src/__init__.py lists it)
E1_HD-209458-b/     experiment 1: HST WFC3, TauREx-III (Secs 4.1, 7)
E2_Hat-P-26b/       experiment 2: JWST NIRSpec G395H, TauREx-III (Secs 4.2, 7.3)
E3_CNN/             experiment 3: HAT-P-26 b, the CNN (Sec. 8); reads E2's simulated observation
figure_code/        one script per figure, written to figure_code/outputs/
```

## Which code is used where in the paper

Each experiment is a numbered list of steps. Each step reads the previous steps' CSVs and writes its own into the
experiment's `outputs/` folder. Figure scripts only read those CSVs and plot.

| Paper | Experiment step | Figures |
|---|---|---|
| Sec. 3, ceteris paribus | none - this is a standalone visualisation | 1 |
| Sec. 5.1, kernel width | E1 step 1 (prior grid), step 4 (analysis) | 2, C1 |
| Sec. 6, dimension reduction | E2 step 1 (prior grid), step 3 (simulated observation), step 4 (analysis) | 3 |
| Sec. 7, HD 209458 b retrievals: 1 + 2 x 108 | E1 step 2 (retrievals) | |
| Sec. 7.1, response and importance | E1 step 4 (analysis) | 4 |
| Sec. 7.2, contributions, global and local metric | E1 step 1 (prior grid), step 3 (posterior forward models), step 4 (analysis) | 5, 6, 7, D1, E1 |
| Sec. 7.2.5, naive vs Gaussian | E1 step 4 (analysis) | 5, F1 |
| Sec. 7.3, HAT-P-26 b retrievals: 1 + 8 | E2 step 2 (retrievals), step 3 (posterior forward models), step 5 (analysis) | 8, 9, 10, 11, H1, I1-I4 |
| Sec. 8 and App. J, the CNN | E3 step 1 (training set), step 2 (training) | |
| Sec. 8.1, CNN posterior | E3 step 3 (posterior estimation)| 12, K1 |
| Sec. 8.2, perturbing the CNN | E3 step 4 (perturbation), step 5 (analysis) | 13, L1 |

## Setup

Create a new conda env

```
conda env create -f environment.yml
conda activate perturb
```

Or without conda: `pip install -r requirements.txt`, and build the MultiNest library yourself
([PyMultiNest install guide](https://johannesbuchner.github.io/PyMultiNest/install.html)).

### Paths

Only `TAUREX_PATH` in `_CONFIG.py` needs editing: replace the placeholder
`/path/to/conda/envs/perturb/bin` with the `bin/` folder of the `perturb` environment. You can print it with:  

```
conda activate perturb && echo "$CONDA_PREFIX/bin"
```

 It looks like `~/miniforge3/envs/perturb/bin`,
`~/miniconda3/envs/perturb/bin`, `~/anaconda3/envs/perturb/bin` or, with Homebrew's Anaconda on macOS,
`/opt/homebrew/anaconda3/envs/perturb/bin` (write it out in full, without `~`). With the pip install it is the
folder that holds the `taurex` command (`dirname "$(which taurex)"`).



 `CROSS_SECTION_PATH` and `CIA_PATH` point at
`DATA/XSEC` and `DATA/CIA` in this repository; change them if you want to use your own set of opacity files from elsewhere.

### Data
The observations and `.par` files are in `DATA/`. The opacities (about 4 GB) are not:
They can be fetched using `./download_data.sh`, or add them manually to the folder. The script will skip any file already in the folder. Keep only one file per gas and per CIA pair in each folder.

| Folder | Files | Source |
|---|---|---|
| `DATA/XSEC` | ExoMol TauREx cross-sections, R = 15 000, 0.3-50 um, one per gas the retrievals fit: H2O (POKAZATEL), CH4 (YT34to10), CO (Li2015), CO2 (HITEMP), NH3 (CoYuTe), SO2 (ExoAmes), C2H2 (aCeTY), C2H4 (MaYTY), H2S (AYT2), HCN (Harris) | [ExoMol opacities](https://www.exomol.com/data/data-types/opacity/) |
| `DATA/CIA` | H2-H2 and H2-He collision-induced absorption | [HITRAN CIA](https://hitran.org/cia/)  |

Note: ExoMol no longer serves the CO2 HITEMP cross-section
(`12C-16O2__HITEMP.R15000_0.3-50mu.xsec.TauREx.h5`), so the script fetches CO2 from UCL-4000 instead; put the HITEMP
file in `DATA/XSEC` before running it to reproduce the paper exactly. 

### Parallel retrievals (MPI)

The environment set up here has no MPI as installation method is different per machine. The paper uses 12 MPI processes.
Either continue without MPI, so each retrieval runs as one process (`N_MPI_PROCESSES = 1`), in which case retrievals will 
take significantly longer, or install an MPI. We recommend Open MPI on macOS. Use 4.1.x, as 5.x fails 
(see [open-mpi/ompi#12273](https://github.com/open-mpi/ompi/issues/12273)). Check with
`mpirun -n 4 python -m mpi4py.bench helloworld`: it must print process 0 to 3 "of 4".

If you are not using macOS, there are other parallelisation options for TauREx (see [TauREx docs](https://taurex3.readthedocs.io/en/latest/user/mpi.html)).
If you are using a cluster, it is probably easiest to stick with whichever version of TauREx is already set up there!


## Full run

run `./run_all.sh`, or single scripts from the
repository root with `PYTHONPATH=.`. The order is experiments 1, 2, 3, then figures: experiment 3
reads experiment 2's simulated observation, and Figs 10, 12 and 13 read both.


## Expected runtime
(Apple M4 Pro, 12 MPI processes)

 **Experiment 1:** about 130 h (217 retrievals of ~36 min)

 **Experiment 2:** about 60 h (9 retrievals of ~6.5 h)

 **Experiment 3:** about 1 h (100 000 forward models, ~15 min of training on the GPU, ~12 min of perturbation on the CPU).


## Figures

| Figure | Script | Experiment |
|---|---|---|
| 1 | `fig01_ceteris_paribus.py` | none |
| 2 | `fig02_correlation_vs_separation.py` | 1 |
| 3 | `fig03_dimension_reduction.py` | 2 |
| 4 | `fig04_hd209_h2o_response.py` | 1 |
| 5 | `fig05_hd209_naive_vs_gaussian.py` | 1 |
| 6 | `fig06_hd209_h2o_importance.py` | 1 |
| 7 | `fig07_hd209_h2o_metrics.py` | 1 |
| 8 | `fig08_hatp26b_perturbations.py` | 2 |
| 9 | `fig09_hatp26b_posteriors.py` | 2 |
| 10 | `fig10_co2_core_correlation.py` | 3 (+2) |
| 11 | `fig11_co2_band_changes.py` | 2 |
| 12 | `fig12_cnn_spectrum.py` | 3 (+2) |
| 13 | `fig13_cnn_response_maps.py` | 3 (+2) |
| C1 | `figC1_hd209_correlation_matrix.py` | 1 |
| D1 | `figD1_hd209_metrics.py` | 1 |
| E1 | `figE1_hd209_importance_all.py` | 1 |
| F1 | `figF1_hd209_fraction_fitted.py` | 1 |
| H1 | `figH1_hatp26b_metrics.py` | 2 |
| I1-I4 | `figI_hatp26b_full_posteriors.py` | 2 |
| K1 | `figK1_cnn_corner.py` | 3 |
| L1 | `figL1_cnn_dropout_masks.py` | 3 |

## Licensing

Code is BSD-3-Clause; data in DATA/ retains its original licences (see the .bib files).
