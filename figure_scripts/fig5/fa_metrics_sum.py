"""
3 comparisons (6 panels)
    Active vs. passive %sv, loading sim, and dimensionality
    for A1 / PEG
"""
import nems0.db as nd

import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
import os
from settings import RESULTS_DIR, BAD_SITES

import scipy.stats as ss
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8 

figpath = "/auto/users/hellerc/code/projects/TBP-ms/figure_files/fig5/"

batch = 324
sqrt = True
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])

sites = [s for s in sites if s not in BAD_SITES]
df = pd.DataFrame(columns=["asv", "psv", "als", "pls", "adim", "pdim", "site", "area", "epoch"])
rr = 0
for site in sites:
    d = pd.read_pickle(os.path.join(RESULTS_DIR, "factor_analysis", str(batch), site, "FA_perstim_PR.pickle"))
    area = nd.pd_query(sql="SELECT area from sCellFile where cellid like %s", params=(f"%{site}%",)).iloc[0][0]
    for e in d["active"].keys():
        df.loc[rr, :] = [
            d["active"][e]["sv"],
            d["passive"][e]["sv"],
            d["active"][e]["loading_sim"],
            d["passive"][e]["loading_sim"],
            d["active"][e]["dim"],
            d["passive"][e]["dim"],
            site,
            area,
            e
        ]
        rr += 1

metrics = ["sv", "ls", "dim"]
for i, m in enumerate(metrics):
    f, ax = plt.subplots(1, 2, figsize=(1.25, 1.1), sharey=True)
    for j, a in enumerate(["A1", "PEG"]):
        mask = df.area==a
        u = df[mask]["p"+m].mean()
        sem = df[mask]["p"+m].std() / np.sqrt(sum(mask))
        ax[j].errorbar([0], [u], yerr=[sem],
                    marker=".", markeredgecolor="k",
                    color="tab:blue", capsize=2, lw=0.5)
        
        u = df[mask]["a"+m].mean()
        sem = df[mask]["a"+m].std() / np.sqrt(sum(mask))
        ax[j].errorbar([1], [u], yerr=[sem],
                    marker=".", markeredgecolor="k",
                    color="tab:orange", capsize=2, lw=0.5)
        ax[j].set_xlim(-0.3, 1.3)
        ax[j].set_xticks([])
        f.tight_layout()
        
        # print stats
        pval = ss.wilcoxon(df[mask]["p"+m], df[mask]["a"+m]).pvalue
        print(f"metric: {m}, area: {a}, pvalue: {pval}")
    # save figure
    f.savefig(os.path.join(figpath, f"fametric_{m}.svg"), dpi=500)