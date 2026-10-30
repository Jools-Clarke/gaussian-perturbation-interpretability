"""One TauREx-III retrieval of a spectrum, run with the taurex program exactly as in the paper (Secs 4 and 7).

retrieve() writes the posterior as CSV: MultiNest samples with their weights. The samples are NOT equally weighted,
so every statistic downstream uses the weights (src.metrics). f of Sec. 2.5 is the weighted posterior mean.
"""
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd

import _CONFIG as config
from src.metrics import weighted_quantile


def _resolve_executable(conda_bin: Path, name: str) -> Path:
    """Return conda_bin/name, raising a clear error if it isn't there."""
    exe = conda_bin / name
    if not exe.exists():
        raise FileNotFoundError(
            f"Could not find '{name}' in conda env bin directory: {conda_bin}. "
            f"Is it installed in this environment?"
        )
    return exe


def _build_env(conda_bin: Path) -> Dict[str, str]:
    """Environment for taurex / mpirun: the processes mpirun spawns must come from the same conda env, not from a
    Homebrew install earlier on PATH."""
    env = os.environ.copy()
    env["PATH"] = f"{conda_bin}{os.pathsep}{env.get('PATH', '')}"
    env["CONDA_PREFIX"] = str(conda_bin.parent)
    env.setdefault("OMPI_MCA_oob_tcp_if_include", "lo0")  # Open MPI on macOS fails with no network otherwise
    # macOS: pymultinest finds libmultinest with ctypes find_library, which does not search the env's lib/
    env.setdefault("DYLD_FALLBACK_LIBRARY_PATH", str(conda_bin.parent / "lib"))
    return env


def _run_streamed(
    cmd: List[str], cwd: Path, env: Dict[str, str], log_path: Path
) -> Tuple[int, float]:
    """Run cmd, printing its output and writing it to log_path. Returns (returncode, wall time in s)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        process.wait()
    elapsed = time.perf_counter() - start

    return process.returncode, elapsed


def extract(path):
    """The posterior in a TauREx .h5, as (posterior, spectra, MAP):
        posterior  one row per MultiNest sample: every fitted and derived parameter, and weight (normalised to sum to 1)
        spectra    wl_um, observed, error, best_fit (TauREx's binned best fit), in ascending wavelength
        MAP        {parameter: value of the maximum a posteriori sample} (NaN for derived parameters)
    """
    with h5py.File(path, "r") as f:
        sol = f["Output/Solutions/solution0"]
        groups = {**{k: sol["fit_params"][k] for k in sol["fit_params"]},
                  **({k: sol["derived_params"][k] for k in sol["derived_params"]} if "derived_params" in sol else {})}
        names = list(groups)
        weight = sol["weights"][()]
        posterior = pd.DataFrame(np.stack([groups[k]["trace"][()] for k in names], axis=1), columns=names)
        posterior["weight"] = weight / weight.sum()
        # TauREx stores spectra in wavenumber order (descending wavelength)
        wl = f["Observed/wlgrid"][()]
        order = np.argsort(wl)
        spectra = pd.DataFrame({"wl_um": wl[order], "observed": f["Observed/spectrum"][()][order],
                                "error": f["Observed/errorbars"][()][order],
                                "best_fit": sol["Spectra/binned_spectrum"][()][order]})
        taurex_median = np.array([float(groups[k]["value"][()]) for k in names])
        best = {k: float(groups[k]["nest_map"][()]) if "nest_map" in groups[k] else np.nan for k in names}
    # our weighted median must reproduce TauREx's own, or the weights are being misread
    ours = np.array([weighted_quantile(posterior[k].values, posterior["weight"].values, 0.5) for k in names])
    assert np.allclose(ours, taurex_median, rtol=1e-3, atol=1e-3), path
    return posterior, spectra, best


def write_spectrum(path, wl, depth, error):
    """TauREx observation file: wavelength (um), (Rp/Rs)^2, error."""
    np.savetxt(path, np.column_stack([wl, depth, error]), fmt="%.9e")


def write_par(template_par, out_par, observation_dat, multinest_dir):
    """Copy of a .par from DATA/ (the settings of Tables B1-B2, G1-G2) with the paths it leaves out added: the data
    paths of _CONFIG.py, the observation and the MultiNest folder. Parsed with configobj, as TauREx does."""
    from configobj import ConfigObj

    par = ConfigObj(str(template_par))
    par["Global"] = {"xsec_path": str(config.CROSS_SECTION_PATH), "cia_path": str(config.CIA_PATH)}
    par["Observation"]["observed_spectrum"] = str(observation_dat)
    par["Optimizer"]["multi_nest_path"] = str(multinest_dir)
    par.filename = str(out_par)
    par.write()


def run_taurex(par, output_h5):
    """taurex -R -i par -o output_h5, under mpirun -n N_MPI_PROCESSES when that is > 1. Both run from the TauREx
    env's bin/ by absolute path: on macOS a Homebrew mpirun on PATH silently runs every rank alone. Returns seconds."""
    conda_bin = Path(config.TAUREX_PATH)
    cmd = [str(_resolve_executable(conda_bin, "taurex")), "-R", "-i", str(par), "-o", str(output_h5)]
    if config.N_MPI_PROCESSES > 1:
        cmd = [str(_resolve_executable(conda_bin, "mpirun")), "-n", str(config.N_MPI_PROCESSES)] + cmd
    log = Path(output_h5).with_suffix(".log")
    returncode, seconds = _run_streamed(cmd, cwd=Path(par).parent, env=_build_env(conda_bin), log_path=log)
    if returncode != 0:
        raise RuntimeError(f"taurex exited with code {returncode}; see {log}")
    return seconds


def retrieve(name, wl, depth, error, par, out_dir):
    """Retrieve one spectrum (ascending wavelength) with the settings in par, and write
        out_dir/posteriors/<name>.csv   one row per MultiNest sample: one column per parameter, plus weight (sums to 1)
        out_dir/spectra/<name>.csv      wl_um, observed, error, best_fit
    TauREx's .dat, .par and .log stay in out_dir/taurex/; its ~150 MB .h5 is deleted once read.
    Returns {"wall_time_s", "map_<param>", ...}: the MAP sample, from which E2 step3 rebuilds the best fit (Fig 11).
    """
    out_dir = Path(out_dir).resolve()  # taurex runs in the run directory, so every path it gets must be absolute
    run = out_dir / "taurex"
    run.mkdir(parents=True, exist_ok=True)
    write_spectrum(run / f"{name}.dat", wl, depth, error)
    with tempfile.TemporaryDirectory() as multinest_dir:
        assert len(multinest_dir) < 90, f"MultiNest truncates paths at 100 characters: set a shorter TMPDIR than {multinest_dir}"
        write_par(par, run / f"{name}.par", run / f"{name}.dat", multinest_dir)
        seconds = run_taurex(run / f"{name}.par", run / f"{name}.h5")
    posterior, spectra, best = extract(run / f"{name}.h5")
    (run / f"{name}.h5").unlink()  # ~150 MB that nothing downstream reads
    for sub, df in (("posteriors", posterior), ("spectra", spectra)):
        (out_dir / sub).mkdir(exist_ok=True)
        df.to_csv(out_dir / sub / f"{name}.csv", index=False)
    return {"wall_time_s": seconds, **{f"map_{p}": v for p, v in best.items()}}
