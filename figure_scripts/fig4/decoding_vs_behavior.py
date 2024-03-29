"""
Compare behavioral vs. neural dprimes
"""
import nems_lbhb.tin_helpers as thelp
import nems0.db as nd

import os
import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from path_helpers import results_file
from settings import RESULTS_DIR, BAD_SITES

import scipy.stats as ss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 10

figpath = "/auto/users/hellerc/code/projects/TBP-ms/figure_files/fig4/"

batch = 324
sqrt = True
sites = [s for s in nd.get_batch_sites(batch)[0] if s not in BAD_SITES]

# load neural dprime
fa = "" #"_FAperstim.1"
amodel = 'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'+fa
pmodel = 'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'+fa
active = []
passive = []
for site in sites:
    try:
        ares = pd.read_pickle(results_file(RESULTS_DIR, site, batch, amodel, "output.pickle"))
        pres = pd.read_pickle(results_file(RESULTS_DIR, site, batch, pmodel, "output.pickle"))
        ares["site"] = site
        pres["site"] = site
        area = nd.pd_query(sql="SELECT area from sCellFile where cellid like %s", params=(f"%{site}%",))
        area = area.iloc[0][0]
        ares["area"] = area
        pres["area"] = area
        if sqrt:
            ares["dp"] = np.sqrt(ares["dp"])
            pres["dp"] = np.sqrt(pres["dp"])
        active.append(ares)
        passive.append(pres)
    except:
        print(f"results not found for site: {site}")
    
active = pd.concat(active)
passive = pd.concat(passive)
df = passive.merge(active, on=["site", "class", "e1", "e2", "area"])
df["delta"] = (df["dp_y"] - df["dp_x"]) / (df["dp_y"] + df["dp_x"])
df["delta_raw"] = df["dp_y"] - df["dp_x"]

# load behavioral dprimes
beh_df = pd.read_pickle(os.path.join(RESULTS_DIR, "behavior_recordings", "all_trials.pickle"))

# Plot relationship between behavior and neural dprime
bg = beh_df.groupby(by=["site", "e1"]).mean()
ng_peg = df[(df["class"]=="tar_cat") & (df.area=="PEG")].groupby(by=["site", "e2"]).mean()
ng_peg.index.set_names("e1", level=1, inplace=True)
ng_peg.index = ng_peg.index.set_levels(ng_peg.index.levels[1].str.strip("TAR_"), level=1)
ng_a1 = df[(df["class"]=="tar_cat") & (df.area=="A1")].groupby(by=["site", "e2"]).mean()
ng_a1.index.set_names("e1", level=1, inplace=True)
ng_a1.index = ng_a1.index.set_levels(ng_a1.index.levels[1].str.strip("TAR_"), level=1)

peg_merge = ng_peg.merge(bg, right_index=True, left_index=True)
a1_merge = ng_a1.merge(bg, right_index=True, left_index=True)

delta_metric = "delta_raw"
nboots = 500
s = 10
delta_ylim = (-1, 3.5)
abs_ylim = (-0.1, 5)
colors = ["grey", "k"]
f, ax = plt.subplots(1, 2, figsize=(4, 2))

for i, (df, c) in enumerate(zip([a1_merge, peg_merge], colors)):

    x = df["dprime"]
    xp = np.linspace(np.min(x), np.max(x), 100)
    # delta dprime
    r, p = ss.pearsonr(x, df[delta_metric])
    leg = f"r={round(r, 3)}, p={round(p, 3)}"
    ax[i].scatter(x, df[delta_metric], 
                    s=s, c=c, edgecolor="none", lw=0)
    # ax[i].set_title(f"{leg}")

    # get line of best fit
    z = np.polyfit(x, df[delta_metric], 1)
    # plot line of best fit
    p_y = z[1] + z[0] * xp
    ax[i].plot(xp, p_y, lw=2, color=c)
    
    # bootstrap condifence interval
    boot_preds = []
    for bb in range(nboots):
        ii = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
        zb = np.polyfit(x[ii], df[delta_metric][ii], 1)
        p_yb = zb[1] + zb[0] * xp
        boot_preds.append(p_yb)
    bse = np.stack(boot_preds).std(axis=0)
    lower = p_y - bse
    upper = p_y + bse
    ax[i].fill_between(xp, lower, upper, color=c, alpha=0.5, lw=0)

    ax[i].set_ylim(delta_ylim)

f.tight_layout()

f.savefig(os.path.join(figpath, "delta_vs_behavior.svg"), dpi=500)

# quantify significance of correlation using bootstrapping
np.random.seed(123)
nboots = 1000
x = a1_merge["dprime"]
rb_a1 = []
for bb in range(nboots):
    ii = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    rb_a1.append(np.corrcoef(x[ii], a1_merge[delta_metric][ii])[0, 1])
x = peg_merge["dprime"]
rb_peg = []
for bb in range(nboots):
    ii = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    rb_peg.append(np.corrcoef(x[ii], peg_merge[delta_metric][ii])[0, 1])

# compute bootstrapped p-values
np.random.seed(123)
nboots = 1000
x = a1_merge["dprime"]
rb_a1_null = []
for bb in range(nboots):
    ii = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    jj = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    rb_a1_null.append(np.corrcoef(x[ii], a1_merge[delta_metric][jj])[0, 1])
x = peg_merge["dprime"]
rb_peg_null = []
for bb in range(nboots):
    ii = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    jj = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    rb_peg_null.append(np.corrcoef(x[ii], peg_merge[delta_metric][jj])[0, 1])
a1_pval = np.mean(np.array(rb_a1_null) > np.corrcoef(a1_merge["dprime"], a1_merge[delta_metric])[0, 1])
peg_pval = np.mean(np.array(rb_peg_null) > np.corrcoef(peg_merge["dprime"], peg_merge[delta_metric])[0, 1])
print(f"A1 pval: {a1_pval}")
print(f"PEG pval: {peg_pval}")

f, ax = plt.subplots(1, 1, figsize=(1, 2))

lower = np.quantile(rb_a1, 0.025)
upper = np.quantile(rb_a1, 0.975)
ax.plot([0, 0], [lower, upper], color="grey", zorder=-1)
ax.scatter([0], [np.mean(rb_a1)], s=50, edgecolor="k", c="grey")

lower = np.quantile(rb_peg, 0.025)
upper = np.quantile(rb_peg, 0.975)
ax.plot([1, 1], [lower, upper], color="k")
ax.scatter([1], [np.mean(rb_peg)], s=50, edgecolor="k", c="k")

ax.axhline(0, linestyle="--", color="grey")

ax.set_xlim((-0.1, 1.1))
ax.set_xticks([])

f.savefig(os.path.join(figpath, "pearson_95conf.svg"), dpi=500)

# Supplement for absolute dprime relationship only
nboots = 100
s = 10
abs_ylim = (-0.1, 5)
f, ax = plt.subplots(1, 2, figsize=(4, 2))

for i, df in enumerate([a1_merge, peg_merge]):

    x = df["dprime"]
    xp = np.linspace(np.min(x), np.max(x), 100)
    # raw dprime
    r, p = ss.pearsonr(x, df["dp_x"])
    leg = f"r={round(r, 3)}, p={round(p, 3)}"
    ax[i].scatter(x, df["dp_x"], 
                    s=s, c="grey", edgecolor="none", lw=0)
    ax[i].set_title(f"{leg}")

    # get line of best fit
    z = np.polyfit(x, df["dp_x"], 1)
    # plot line of best fit
    p_y = z[1] + z[0] * xp
    ax[i].plot(xp, p_y, lw=2, color="k")
    
    boot_preds = []
    for bb in range(nboots):
        ii = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
        zb = np.polyfit(x[ii], df["dp_x"][ii], 1)
        p_yb = zb[1] + zb[0] * xp
        boot_preds.append(p_yb)
    bse = np.stack(boot_preds).std(axis=0)
    lower = p_y - bse
    upper = p_y + bse
    ax[i].fill_between(xp, lower, upper, color="k", alpha=0.5, lw=0)

    ax[i].set_ylim(abs_ylim)

f.tight_layout()