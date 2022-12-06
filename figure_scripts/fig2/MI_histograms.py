import os
import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from settings import RESULTS_DIR, BAD_SITES

import scipy.stats as ss
import nems_lbhb.tin_helpers as thelp
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8
mpl.rcParams['xtick.labelsize'] = 8 
mpl.rcParams['ytick.labelsize'] = 8 


figpath = "/auto/users/hellerc/code/projects/TBP-ms/figure_files/fig2/"

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


# A1 HISTOGRAMS
f, ax = plt.subplots(4, 1, figsize=(1.5, 3))

for i, snr in enumerate(snrs[::-1]):
    # A1
    mask = (df.snr==snr) & (df.area=="A1")
    weights = np.ones_like(df["MI"][mask]) / len(df["MI"][mask])
    ax[i].hist(df["MI"][mask], bins=bins,
        histtype="stepfilled", edgecolor="k", color=gR(i),
        weights=weights)
    # print snr / pvalue
    pval = ss.wilcoxon(df["MI"][mask].values).pvalue
    print(f"SNR: {snr}, n: {sum(mask)}, pval: {pval}")
for a in ax.flatten():
    a.axvline(0, linestyle="--", color="k")
    a.set_xticks([])
    a.set_ylim((0, 0.3))

f.tight_layout()

f.savefig(os.path.join(figpath, "A1_MI_histograms.svg"), dpi=500)

# PEG HISTOGRAMS
f, ax = plt.subplots(4, 1, figsize=(1.5, 3))

for i, snr in enumerate(snrs[::-1]):
    # A1
    mask = (df.snr==snr) & (df.area=="PEG")
    weights = np.ones_like(df["MI"][mask]) / len(df["MI"][mask])
    ax[i].hist(df["MI"][mask], bins=bins,
        histtype="stepfilled", edgecolor="k", color=gR(i),
        weights=weights)
    # print snr / pvalue
    pval = ss.wilcoxon(df["MI"][mask].values).pvalue
    print(f"SNR: {snr}, n: {sum(mask)}, pval: {pval}")

for a in ax.flatten():
    a.axvline(0, linestyle="--", color="k")
    a.set_xticks([])
    a.set_ylim((0, 0.3))

f.tight_layout()

f.savefig(os.path.join(figpath, "PEG_MI_histograms.svg"), dpi=500)