"""
Illustrate change in loading similarity with a single
site. Then have a summary panel.
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
    d = pd.read_pickle(os.path.join(RESULTS_DIR, "factor_analysis", str(batch), site, "FA_perstim.pickle"))
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

# plot loading weights for an example site
site = "JLY031c"
d = pd.read_pickle(os.path.join(RESULTS_DIR, "factor_analysis", str(batch), site, "FA_perstim.pickle"))
s = [s for s in d["active"].keys() if "CAT_" in s][0]
ncells = d["active"][s]["components_"].shape[1]

f, ax = plt.subplots(1, 1, figsize=(3, 2))

markerline, stemlines, baseline = ax.stem(
    np.arange(0, ncells)-0.3, d["passive"][s]["components_"][0, :], 
                        markerfmt=".")
markerline.set_markerfacecolor('tab:blue')
stemlines.set_color("tab:blue")
stemlines.set_linewidth(0.5)
baseline.set_color("k")
markerline, stemlines, baseline = ax.stem(
    np.arange(0, ncells)+0.3, d["active"][s]["components_"][0, :], 
                        markerfmt=".")
markerline.set_markerfacecolor('tab:orange')
stemlines.set_color("tab:orange")
stemlines.set_linewidth(0.5)
baseline.set_color("k")

f.tight_layout()
f.savefig(os.path.join(figpath, "loading_weights.svg"), dpi=500)

ncells = d["passive"][s]["components_"].shape[1]
f, ax = plt.subplots(1, 1, figsize=(1, 2))

ax.scatter(np.zeros(ncells), d["passive"][s]["components_"][0, :], 
                    s=25, edgecolor="none", alpha=0.5)
ax.scatter(np.ones(ncells), d["active"][s]["components_"][0, :], 
                    s=25, edgecolor="none", alpha=0.5)
ax.set_xlim((-0.5, 1.5))

f.tight_layout()
f.savefig(os.path.join(figpath, "loading_weights_scatter.svg"), dpi=500)

f, ax = plt.subplots(1, 2, figsize=(2, 2), sharey=True)
m = "ls"
for j, a in enumerate(["A1", "PEG"]):
    mask = df.area==a
    u = df[mask]["p"+m].mean()
    sem = df[mask]["p"+m].std() / np.sqrt(sum(mask))
    ax[j].errorbar([0], [u], yerr=[sem],
                marker="o", markeredgecolor="k",
                color="tab:blue", capsize=2, lw=2)
    
    u = df[mask]["a"+m].mean()
    sem = df[mask]["a"+m].std() / np.sqrt(sum(mask))
    ax[j].errorbar([1], [u], yerr=[sem],
                marker="o", markeredgecolor="k",
                color="tab:orange", capsize=2, lw=2)
    ax[j].set_xlim(-0.3, 1.3)
    ax[j].set_xticks([])
    f.tight_layout()
    
    # print stats
    pval = ss.wilcoxon(df[mask]["p"+m], df[mask]["a"+m]).pvalue
    print(f"metric: {m}, area: {a}, pvalue: {pval}")
# save figure
f.savefig(os.path.join(figpath, f"fametric_{m}.svg"), dpi=500)