"""
Summary of behavior performance across all training (on real task)
for each animal
"""
import statistics
import json
import nems_lbhb.tin_helpers as thelp
import pickle
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.behavior import get_reaction_times
from nems_lbhb.behavior_plots import plot_RT_histogram
import nems0.db as nd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 12

results_path = "/auto/users/hellerc/results/TBP-ms/behavior"

animals = [
    "Armillaria",
    "Cordyceps",
    "Jellybaby",
    "Clathrus"
]

# plot RT histogram for each animal independently
# save d-prime / DI per session for summary psychometric
dp = {}
di = {}
for an in animals:
    results = json.load(open( os.path.join(results_path, f"{an}_training.json"), 'r' ) )

    # make a "dummy" freq for builing targets list
    rts = results["RTs"]
    dprime = results["dprime"]
    DI = results["DI"]
    nsessions = results["n"]
    keep = [k for k, v in nsessions.items() if (v > 10) | (k == '-inf')]
    rts = {k: v for k, v in rts.items() if k in keep}
    dprime = {k: v for k, v in dprime.items() if k in keep}
    DI = {k: v for k, v in DI.items() if k in keep}
    nsessions = {k: v for k, v in nsessions.items() if k in keep}

    targets = [k for k in rts.keys()]
    targets = [("300+"+t+"+Noise").replace("inf", "Inf") for t in targets]
    cat = targets[0]
    BwG, gR = thelp.make_tbp_colormaps([cat], targets, use_tar_freq_idx=0)
    legend = [s+ f" dB, n={nsessions[s]}, d'={round(np.mean(dprime[s]), 3)}" if '-inf' not in s else 'Catch' for s in rts.keys()]

    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    bins = np.arange(0, 1.4, 0.001)
    plot_RT_histogram(rts, bins=bins, ax=ax, cmap=gR, lw=2, legend=legend)
    ax.set_title(an)
    f.tight_layout()

    # save di / dprime
    di[an] = DI
    dp[an] = dprime


# plot mean / se for each animal
kk = ["-10", "-5", "0", "inf"]
xx = [0, 1, 2, 3]
f, ax = plt.subplots(1, 1, figsize=(5, 5))

for an in animals[::-1]:
    dprime = dp[an]
    yy = np.zeros(len(xx))
    yye = np.zeros(len(xx))
    for (i, k) in enumerate(kk):
        if k in dprime.keys():
            yy[i] = np.mean(dprime[k])
            yye[i] = np.std(dprime[k]) / np.sqrt(len(dprime[k]))
        else:
            yy[i] = np.nan
            yye[i] = np.nan

    ff = np.isnan(yy)==False
    # ax.errorbar(np.array(xx)[ff], yy[ff], yerr=yye[ff], 
    #             capsize=2, marker="o", lw=2, label=an)
    ax.plot(np.array(xx)[ff], yy[ff], ".-", lw=2, label=an)
    ax.fill_between(np.array(xx)[ff], yy[ff]+yye[ff], yy[ff]-yye[ff],
            color=ax.get_lines()[-1].get_color(),
            alpha=0.3, lw=0)

ax.axhline(0, linestyle="--", color="k")
ax.set_xticks(xx)
ax.set_xticklabels(kk)
ax.set_xlabel("SNR")
ax.set_ylabel(r"$d'$")
ax.legend(frameon=False, bbox_to_anchor=(0, 1), loc="upper left")

