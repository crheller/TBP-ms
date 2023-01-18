"""
2x2
top row, mean tar vs. cat response for active/passive, each neuron
bottom row:
    active vs. passive d' scatter plot
    delta dprime histogram
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8
mpl.rcParams['xtick.labelsize'] = 8 
mpl.rcParams['ytick.labelsize'] = 8 

figpath = "/auto/users/hellerc/code/projects/TBP-ms/figure_files/fig2/"

df_resp = pd.read_csv("/auto/users/hellerc/results/TBP-ms/tar_vs_cat.csv", index_col=0)
df_dprime = pd.read_csv("/auto/users/hellerc/results/TBP-ms/singleNeuronDprime.csv", index_col=0)
gg_resp = df_resp.groupby(by=["snr", "cellid", "area"]).mean()
s = 3
alpha = 0.7


# PEG
ggm = gg_resp[gg_resp.index.get_level_values(2)=="PEG"]
yvals = ggm[ggm.index.get_level_values(0)=="InfdB"]
xvals = ggm[ggm.index.get_level_values(0)=="-InfdB"]
vals = yvals.merge(xvals, on="cellid")

# mean responses passive
f, ax = plt.subplots(1, 1, figsize=(1, 1))

ax.scatter(
    vals["passive_y"],
    vals["passive_x"],
    s=s, c="k", alpha=alpha,
    edgecolor="none"
)
ax.set_xlim((vals.values.min(), vals.values.max()))
ax.set_ylim((vals.values.min(), vals.values.max()))
ax.plot([vals.values.min(), vals.values.max()],
            [vals.values.min(), vals.values.max()], 
            "grey", linestyle="--", zorder=-1)
f.savefig(os.path.join(figpath, "passive_resp_peg.svg"), dpi=500)

# mean responses active
f, ax = plt.subplots(1, 1, figsize=(1, 1))
ax.scatter(
    vals["active_y"],
    vals["active_x"],
    s=s, c="k", alpha=alpha,
    edgecolor="none"
)
ax.set_xlim((vals.values.min(), vals.values.max()))
ax.set_ylim((vals.values.min(), vals.values.max()))
ax.plot([vals.values.min(), vals.values.max()],
            [vals.values.min(), vals.values.max()], 
            "grey", linestyle="--", zorder=-1)
f.savefig(os.path.join(figpath, "active_resp_peg.svg"), dpi=500)

# dprime scatter
dd = df_dprime[(df_dprime.area=="PEG") & (df_dprime.category=="tar_cat")]
dd = dd[(dd.e1.str.contains("\+InfdB")) | (dd.e2.str.contains("\+InfdB"))]
f, ax = plt.subplots(1, 1, figsize=(1, 1))

ax.scatter(
    dd["passive"], dd["active"],
    s=s, alpha=alpha, color="k",
    edgecolor="none"
)
ax.plot([0, 7.5], [0, 7.5], "grey", linestyle="--")
ax.set_xlim((0, 7.5))
ax.set_ylim((0, 7.5))
ax.set_aspect("equal")
f.savefig(os.path.join(figpath, "dprime_peg.svg"), dpi=500)

# delta dprime histogram
f, ax = plt.subplots(1, 1, figsize=(1, 1))

ax.hist(
    dd["active"]-dd["passive"],
    bins=np.arange(-3, 3.2, 0.2),
    facecolor="lightgrey",
    edgecolor="k",
    histtype="stepfilled"
)
f.savefig(os.path.join(figpath, "delta_dprime_peg.svg"), dpi=500)



# A1
ggm = gg_resp[gg_resp.index.get_level_values(2)=="A1"]
yvals = ggm[ggm.index.get_level_values(0)=="InfdB"]
xvals = ggm[ggm.index.get_level_values(0)=="-InfdB"]
vals = yvals.merge(xvals, on="cellid")

# mean responses passive
f, ax = plt.subplots(1, 1, figsize=(1, 1))

ax.scatter(
    vals["passive_y"],
    vals["passive_x"],
    s=s, c="k", alpha=alpha,
    edgecolor="none"
)
ax.set_xlim((vals.values.min(), vals.values.max()))
ax.set_ylim((vals.values.min(), vals.values.max()))
ax.plot([vals.values.min(), vals.values.max()],
            [vals.values.min(), vals.values.max()], 
            "grey", linestyle="--", zorder=-1)
f.savefig(os.path.join(figpath, "passive_resp_a1.svg"), dpi=500)

# mean responses active
f, ax = plt.subplots(1, 1, figsize=(1, 1))
ax.scatter(
    vals["active_y"],
    vals["active_x"],
    s=s, c="k", alpha=alpha,
    edgecolor="none"
)
ax.set_xlim((vals.values.min(), vals.values.max()))
ax.set_ylim((vals.values.min(), vals.values.max()))
ax.plot([vals.values.min(), vals.values.max()],
            [vals.values.min(), vals.values.max()], 
            "grey", linestyle="--", zorder=-1)
f.savefig(os.path.join(figpath, "active_resp_a1.svg"), dpi=500)

# dprime scatter
dd = df_dprime[(df_dprime.area=="A1") & (df_dprime.category=="tar_cat")]
dd = dd[(dd.e1.str.contains("\+InfdB")) | (dd.e2.str.contains("\+InfdB"))]
f, ax = plt.subplots(1, 1, figsize=(1, 1))

ax.scatter(
    dd["passive"], dd["active"],
    s=s, alpha=alpha, color="k",
    edgecolor="none"
)
ax.plot([0, 7.5], [0, 7.5], "grey", linestyle="--")
ax.set_xlim((0, 7.5))
ax.set_ylim((0, 7.5))
ax.set_aspect("equal")
f.savefig(os.path.join(figpath, "dprime_a1.svg"), dpi=500)

# delta dprime histogram
f, ax = plt.subplots(1, 1, figsize=(1, 1))

ax.hist(
    dd["active"]-dd["passive"],
    bins=np.arange(-3, 3.2, 0.2),
    facecolor="lightgrey",
    edgecolor="k",
    histtype="stepfilled"
)
f.savefig(os.path.join(figpath, "delta_dprime_a1.svg"), dpi=500)