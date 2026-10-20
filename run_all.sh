#!/usr/bin/env bash
# Regenerate every figure in the paper from scratch: three experiments, then the figure scripts.
# Every step always recomputes its outputs (into its experiment's outputs/ folder); there is no caching. Run from anywhere.
# About a week on a 12-core laptop with MPI (README, "Expected runtime"), almost all of it the TauREx-III retrievals.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

# Experiment 1: HD 209458 b, TauREx-III (~130 h)
python E1_HD-209458-b/step1_forward_models.py
python E1_HD-209458-b/step2_retrievals.py
python E1_HD-209458-b/step3_posterior_forward_models.py
python E1_HD-209458-b/step4_analysis.py

# Experiment 2: HAT-P-26 b, TauREx-III (~60 h)
python E2_Hat-P-26b/step1_forward_models.py
python E2_Hat-P-26b/step2_retrievals.py
python E2_Hat-P-26b/step3_posterior_forward_models.py
python E2_Hat-P-26b/step4_dimension_reduction.py
python E2_Hat-P-26b/step5_analysis.py

# Experiment 3: HAT-P-26 b, CNN (needs experiment 2's simulated observation)
python E3_CNN/step1_training_set.py
python E3_CNN/step2_train.py
python E3_CNN/step3_posterior.py
python E3_CNN/step4_perturbation.py
python E3_CNN/step5_analysis.py

# Figures: plotting only, each reads CSVs from E*/outputs/ and writes figure_code/outputs/
for f in figure_code/fig*.py; do
    python "$f"
done
