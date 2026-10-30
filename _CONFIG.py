"""Machine paths, and the settings shared between experiments. Settings used by one script live at its top."""
from pathlib import Path as _Path

# Machine paths (README, "Paths"). bin/ of the environment.yml env, i.e. $CONDA_PREFIX/bin with it active: taurex and
# mpirun run from here (src/retrieval.py)
TAUREX_PATH = "/path/to/conda/envs/perturb/bin"

# HITRAN collision-induced absorption (TauREx [Global] cia_path), filled by download_data.sh
CIA_PATH = _Path(__file__).resolve().parent / "DATA" / "CIA"

# molecular cross-sections (TauREx [Global] xsec_path), filled by download_data.sh
CROSS_SECTION_PATH = _Path(__file__).resolve().parent / "DATA" / "XSEC"

# mpirun -n for the retrievals (12 in the paper); 1 runs taurex without MPI (README, "Parallel retrievals")
N_MPI_PROCESSES = 1

# processes for the batches of forward models (E1 step3, E2 step3, E3 steps 1 and 3)
N_WORKERS = 10

# Experiment 1: the observation and retrieval settings (Sec. 4.1, Tables B1, B2)
HD209_PAR = _Path(__file__).resolve().parent / "DATA" / "HD209458b" / "HD209458b-Al-Rafaie-Taurex-benchmark.par"
HD209_OBS = _Path(__file__).resolve().parent / "DATA" / "HD209458b" / "HD209458b-Tsarias-WFC3.dat"

# Experiment 2: the same (Sec. 4.2, Tables G1, G2); experiment 3 reads HATP26B_OBS too
HATP26B_PAR = _Path(__file__).resolve().parent / "DATA" / "HAT-P-26b" / "hatp26b_taurex_transitspectroscopy.par"
HATP26B_OBS = _Path(__file__).resolve().parent / "DATA" / "HAT-P-26b" / "HAT-P-26b_NIRSpec_G395H.dat"

# FWHM of the Gaussian window in um, all three experiments (Sec. 5.1: 0.101 um from the Fig 2 fit)
GAUSSIAN_FWHM_UM = 0.1

# The narrow HAT-P-26 b bands that experiments 2 and 3 target (Sec. 7.3): gas -> band centre (um)
TARGET_BANDS = {"SO2": 4.041, "CO2": 4.293}

# HAT-P-26 b posterior means (Table G3): the simulated observation (E2 step3, Secs 6 and 8.1) is the forward model at
# them, and T, Rp and the cloud top stay fixed at them in the CNN's training set (E3 step1)
HATP26B_LOG_VMR = {
    "H2O": -1.63,
    "CO2": -2.77,
    "SO2": -4.78,
    "NH3": -8.89,
    "CO": -7.73,
    "CH4": -8.38,
    "C2H2": -9.31,
    "C2H4": -8.70,
    "H2S": -4.91,
    "HCN": -8.94,
}
FIXED_PARAMS = {  # TauREx parameter names
    "T": 605.0,  # K, isothermal
    "planet_radius": 0.5431,  # R_jup
    "clouds_pressure": 10**3.85,  # Pa
}

# The ten gases of the HAT-P-26 b retrieval (Table G2), in the CNN's output order (experiments 2 and 3)
MOLECULES = ["H2O", "CO2", "SO2", "H2S", "CO", "CH4", "NH3", "HCN", "C2H2", "C2H4"]
# Experiment 3: seed of the training draws, data split, validation noise, posterior noise and dropout masks
SEED = 26
