"""Code shared by the experiments and figures.

    perturbation.py    the method: naive and Gaussian perturbation (Eqs. 1-3, 9, 11) and the importance (Eq. 13)
    metrics.py         weighted posterior statistics, and the global and local metrics (Eqs. 14, 15)
    forward_model.py   TauREx-III forward models built from the retrieval .par files in DATA/
    retrieval.py       one TauREx-III retrieval, its weighted posterior saved as CSV
    cnn.py             the CNN retrieval of Sec. 8 and its Monte Carlo dropout black box f
    plotting.py        plot style and helpers for figure_code/
"""
