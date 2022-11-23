import os
import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from settings import RESULTS_DIR, BAD_SITES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

resultspath = os.path.join(RESULTS_DIR, "modulation_index.csv")

df = pd.read_csv(resultspath, index_col=0)

snrs = [np.inf, 0, -5, -np.inf]
bins = np.arange(-1, 1.1, 0.1)
f, ax = plt.subplots(2, 4, figsize=(8, 4))

for i, snr in enumerate(snrs):
    # A1
    mask = (df.snr==snr) & (df.area=="A1")
    ax[0, i].hist(df["MI"][mask], bins=bins,
        histtype="stepfilled", edgecolor="k", color="lightgrey")
    # PEG
    mask = (df.snr==snr) & (df.area=="PEG")
    ax[1, i].hist(df["MI"][mask], bins=bins,
        histtype="stepfilled", edgecolor="k", color="lightgrey")

for a in ax.flatten():
    a.axvline(0, linestyle="--", color="k")