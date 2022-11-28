import os
import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from settings import RESULTS_DIR, BAD_SITES

import nems_lbhb.tin_helpers as thelp
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

N_tar = 4
mid_gray = 210/256
vals = np.ones((N_tar, 4))
vals[:, 0] = np.linspace(mid_gray, 1, N_tar)
vals[:, 1] = np.linspace(mid_gray, 1-mid_gray, N_tar)
vals[:, 2] = np.linspace(mid_gray, 1-mid_gray, N_tar)
gR = ListedColormap(vals, 'gR')

resultspath = os.path.join(RESULTS_DIR, "modulation_index.csv")

df = pd.read_csv(resultspath, index_col=0)

snrs = [np.inf, 0, -5, -np.inf]
bins = np.arange(-1, 1.1, 0.1)

f, ax = plt.subplots(2, 4, figsize=(8, 4))

for i, snr in enumerate(snrs[::-1]):
    # A1
    mask = (df.snr==snr) & (df.area=="A1")
    ax[0, i].hist(df["MI"][mask], bins=bins,
        histtype="stepfilled", edgecolor="k", color=gR(i))
    ax[0, i].set_title(f"A1, {snr}")
    # PEG
    mask = (df.snr==snr) & (df.area=="PEG")
    ax[1, i].hist(df["MI"][mask], bins=bins,
        histtype="stepfilled", edgecolor="k", color=gR(i))
    ax[1, i].set_title(f"PEG, {snr}")

for a in ax.flatten():
    a.axvline(0, linestyle="--", color="k")

f.tight_layout()