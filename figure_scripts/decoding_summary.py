"""
Example ellipse plots for A1 / PEG, active / passive
Summary of active vs. passive d' per category
x=y project of delta dprimes -- selective change in PEG
"""
import charlieTools.TBP_ms.loaders as loaders
import charlieTools.plotting as cplt
import nems_lbhb.tin_helpers as thelp
import nems0.db as nd

import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from path_helpers import results_file
from settings import RESULTS_DIR, BAD_SITES

import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

batch = 324
sqrt = True
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])

# load decoding results
amodel = 'tbpDecoding_mask.h.cr.m_decmask.h.cr.m.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise-sharedSpace'
pmodel = 'tbpDecoding_mask.pa_decmask.h.cr.m.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise-sharedSpace'
fmodel = 'tbpDecoding_mask.pa_decmask.h.cr.m.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise-sharedSpace'
sites = [s for s in sites if s not in BAD_SITES]
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
df_drop = df[["dp_x", "dp_y", "e1", "e2", "site"]].drop_duplicates()
df = df.loc[df_drop.index]
df["delta"] = (df["dp_y"] - df["dp_x"]) / (df["dp_y"] + df["dp_x"])
df["delta_raw"] = df["dp_y"] - df["dp_x"]

# load data from two example sites (one A1, one PEG)
a1_site = "CRD016c"
a1p, _ = loaders.load_tbp_for_decoding(site=a1_site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=["PASSIVE_EXPERIMENT"],
                                    recache=False)
a1a, _ = loaders.load_tbp_for_decoding(site=a1_site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=["HIT_TRIAL", "CORRECT_REJECT_TRIAL", "MISS_TRIAL"],
                                    recache=False)
peg_site = "CRD010b"
pegp, _ = loaders.load_tbp_for_decoding(site=peg_site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=["PASSIVE_EXPERIMENT"],
                                    recache=False)
pega, _ = loaders.load_tbp_for_decoding(site=peg_site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=["HIT_TRIAL", "CORRECT_REJECT_TRIAL", "MISS_TRIAL"],
                                    recache=False)

# get (fixed) dDR space for each site
dd = pd.read_pickle(results_file(RESULTS_DIR, a1_site, batch, fmodel, "output.pickle"))
a1_loading = dd["dr_loadings"].iloc[0]
dd = pd.read_pickle(results_file(RESULTS_DIR, peg_site, batch, fmodel, "output.pickle"))
peg_loading = dd["dr_loadings"].iloc[0]

## Plot ellipse plots
s = 10
f, ax = plt.subplots(2, 2, figsize=(5, 5))

# A1
catch = [c for c in a1a.keys() if "CAT_" in c]
targets = catch+[t for t in a1a.keys() if "TAR_" in t]
BwG, gR = thelp.make_tbp_colormaps(catch, targets, use_tar_freq_idx=0)
for i, t in enumerate(targets):
    xx = a1p[t][:, :, 0].T.dot(a1_loading.T)
    ax[0, 0].scatter(xx[:, 0], xx[:, 1], c=gR(i), s=s)
    el = cplt.compute_ellipse(xx[:, 0], xx[:, 1])
    ax[0, 0].plot(el[0], el[1], c=gR(i), lw=2)
for i, t in enumerate(targets):
    xx = a1a[t][:, :, 0].T.dot(a1_loading.T)
    ax[0, 1].scatter(xx[:, 0], xx[:, 1], c=gR(i), s=s)
    el = cplt.compute_ellipse(xx[:, 0], xx[:, 1])
    ax[0, 1].plot(el[0], el[1], c=gR(i), lw=2)
xmm = np.percentile(ax[0, 1].get_xlim()+ax[0, 0].get_xlim(), [0, 100])
ymm = np.percentile(ax[0, 1].get_ylim()+ax[0, 0].get_ylim(), [0, 100])
ax[0, 0].set_xlim(xmm); ax[0, 1].set_xlim(xmm)
ax[0, 0].set_ylim(ymm); ax[0, 1].set_ylim(ymm)

# PEG
catch = [c for c in pega.keys() if "CAT_" in c]
targets = catch+[t for t in pega.keys() if "TAR_" in t]
BwG, gR = thelp.make_tbp_colormaps(catch, targets, use_tar_freq_idx=0)
for i, t in enumerate(targets):
    xx = pegp[t][:, :, 0].T.dot(peg_loading.T)
    ax[1, 0].scatter(xx[:, 0], xx[:, 1], c=gR(i), s=s)
    el = cplt.compute_ellipse(xx[:, 0], xx[:, 1])
    ax[1, 0].plot(el[0], el[1], c=gR(i), lw=2)
for i, t in enumerate(targets):
    xx = pega[t][:, :, 0].T.dot(peg_loading.T)
    ax[1, 1].scatter(xx[:, 0], xx[:, 1], c=gR(i), s=s)
    el = cplt.compute_ellipse(xx[:, 0], xx[:, 1])
    ax[1, 1].plot(el[0], el[1], c=gR(i), lw=2)
xmm = np.percentile(ax[1, 1].get_xlim()+ax[1, 0].get_xlim(), [0, 100])
ymm = np.percentile(ax[1, 1].get_ylim()+ax[1, 0].get_ylim(), [0, 100])
ax[1, 0].set_xlim(xmm); ax[1, 1].set_xlim(xmm)
ax[1, 0].set_ylim(ymm); ax[1, 1].set_ylim(ymm)

for a in ax.flatten():
    a.set_xlabel(r"$\Delta \mu$")
    a.set_ylabel(r"$\sigma$")
    a.set_xticks([])
    a.set_yticks([])

f.patch.set_facecolor("white")
f.tight_layout()

## Scatter plot, group by site
f, ax = plt.subplots(1, 2, figsize=(6, 3))

ax[0].scatter(
    df[(df["class"]=="tar_tar") & (df.area=="A1")].groupby(by=["site", "area"]).mean()["dp_x"],
    df[(df["class"]=="tar_tar") & (df.area=="A1")].groupby(by=["site", "area"]).mean()["dp_y"],
    facecolor="r", edgecolor="k", s=50
)
ax[0].scatter(
    df[(df["class"]=="tar_cat") & (df.area=="A1")].groupby(by=["site", "area"]).mean()["dp_x"],
    df[(df["class"]=="tar_cat") & (df.area=="A1")].groupby(by=["site", "area"]).mean()["dp_y"],
    facecolor="grey", edgecolor="k", s=50, zorder=-1
)
# ref / ref
ax[0].scatter(
    df[(df["class"]=="ref_ref") & (df.area=="A1")].groupby(by=["site", "area"]).mean()["dp_x"],
    df[(df["class"]=="ref_ref") & (df.area=="A1")].groupby(by=["site", "area"]).mean()["dp_y"],
    facecolor="blue", edgecolor="k", s=50
)
mm = np.min(ax[0].get_xlim() + ax[0].get_ylim())
m = np.max(ax[0].get_xlim() + ax[0].get_ylim())
ax[0].plot([mm, m], [mm, m], "k--")
ax[0].set_title("A1")

ax[1].scatter(
    df[(df["class"]=="tar_tar") & (df.area=="PEG")].groupby(by=["site", "area"]).mean()["dp_x"],
    df[(df["class"]=="tar_tar") & (df.area=="PEG")].groupby(by=["site", "area"]).mean()["dp_y"],
    facecolor="r", edgecolor="k", s=50
)
ax[1].scatter(
    df[(df["class"]=="tar_cat") & (df.area=="PEG")].groupby(by=["site", "area"]).mean()["dp_x"],
    df[(df["class"]=="tar_cat") & (df.area=="PEG")].groupby(by=["site", "area"]).mean()["dp_y"],
    facecolor="grey", edgecolor="k", s=50
)
# ref / ref
ax[1].scatter(
    df[(df["class"]=="ref_ref") & (df.area=="PEG")].groupby(by=["site", "area"]).mean()["dp_x"],
    df[(df["class"]=="ref_ref") & (df.area=="PEG")].groupby(by=["site", "area"]).mean()["dp_y"],
    facecolor="blue", edgecolor="k", s=50, zorder=-1
)
mm = np.min(ax[1].get_xlim() + ax[1].get_ylim())
m = np.max(ax[1].get_xlim() + ax[1].get_ylim())
ax[1].plot([mm, m], [mm, m], "k--")
ax[1].set_title("PEG")

for a in ax:
    a.set_ylabel(r"Active $d'$")
    a.set_xlabel(r"Passive $d'$")

f.tight_layout()

## Delta dprime strip plot
s = 10
delta_metric = "delta"
f, ax = plt.subplots(1, 2, figsize=(3, 3), sharey=True)

# A1
xx = np.random.normal(0, 0.03, len(df[(df.area=="A1") & (df["class"]=="tar_cat")].site.unique()))
ax[0].scatter(
    xx,
    df[(df["class"]=="tar_tar") & (df.area=="A1")].groupby(by=["site", "area"]).mean()[delta_metric],
    s=s, c="r"
)
u = df[(df["class"]=="tar_tar") & (df.area=="A1")].groupby(by=["site", "area"]).mean()[delta_metric].mean()
yerr = df[(df["class"]=="tar_tar") & (df.area=="A1")].groupby(by=["site", "area"]).mean()[delta_metric].std() / np.sqrt(len(xx))
ax[0].errorbar(0, u, yerr=yerr, marker="o", 
            capsize=2, lw=1, markerfacecolor="r", markeredgecolor="k", color="k")     
xx += 1
ax[0].scatter(
    xx,
    df[(df["class"]=="tar_cat") & (df.area=="A1")].groupby(by=["site", "area"]).mean()[delta_metric],
    s=s, c="grey"
)
u = df[(df["class"]=="tar_cat") & (df.area=="A1")].groupby(by=["site", "area"]).mean()[delta_metric].mean()
yerr = df[(df["class"]=="tar_cat") & (df.area=="A1")].groupby(by=["site", "area"]).mean()[delta_metric].std() / np.sqrt(len(xx))
ax[0].errorbar(1, u, yerr=yerr, marker="o", 
            capsize=2, lw=1, markerfacecolor="grey", markeredgecolor="k", color="k") 
# ref / ref
xx += 1
ax[0].scatter(
    xx,
    df[(df["class"]=="ref_ref") & (df.area=="A1")].groupby(by=["site", "area"]).mean()[delta_metric],
    s=s, c="blue"
)
u = df[(df["class"]=="ref_ref") & (df.area=="A1")].groupby(by=["site", "area"]).mean()[delta_metric].mean()
yerr = df[(df["class"]=="ref_ref") & (df.area=="A1")].groupby(by=["site", "area"]).mean()[delta_metric].std() / np.sqrt(len(xx))
ax[0].errorbar(2, u, yerr=yerr, marker="o", 
            capsize=2, lw=1, markerfacecolor="blue", markeredgecolor="k", color="k") 

for site in df[df.area=="A1"].site.unique():
    x1 = df[(df.site==site) & (df["class"]=="tar_tar")][delta_metric].mean()
    x2 = df[(df.site==site) & (df["class"]=="tar_cat")][delta_metric].mean()
    x3 = df[(df.site==site) & (df["class"]=="ref_ref")][delta_metric].mean()
    ax[0].plot([0.2, 0.8], [x1, x2], color="k", lw=0.5)
    ax[0].plot([1.2, 1.8], [x2, x3], color="k", lw=0.5)

ax[0].set_title("A1")
# PEG
xx = np.random.normal(0, 0.03, len(df[(df.area=="PEG") & (df["class"]=="tar_tar")].site.unique()))
ax[1].scatter(
    xx,
    df[(df["class"]=="tar_tar") & (df.area=="PEG")].groupby(by=["site", "area"]).mean()[delta_metric],
    s=s, c="r"
)
u = df[(df["class"]=="tar_tar") & (df.area=="PEG")].groupby(by=["site", "area"]).mean()[delta_metric].mean()
yerr = df[(df["class"]=="tar_tar") & (df.area=="PEG")].groupby(by=["site", "area"]).mean()[delta_metric].std() / np.sqrt(len(xx))
ax[1].errorbar(0, u, yerr=yerr, marker="o", 
            capsize=2, lw=1, markerfacecolor="r", markeredgecolor="k", color="k")     
xx = np.random.normal(1, 0.03, len(df[(df.area=="PEG") & (df["class"]=="tar_cat")].site.unique()))
ax[1].scatter(
    xx,
    df[(df["class"]=="tar_cat") & (df.area=="PEG")].groupby(by=["site", "area"]).mean()[delta_metric],
    s=s, c="grey"
)
u = df[(df["class"]=="tar_cat") & (df.area=="PEG")].groupby(by=["site", "area"]).mean()[delta_metric].mean()
yerr = df[(df["class"]=="tar_cat") & (df.area=="PEG")].groupby(by=["site", "area"]).mean()[delta_metric].std() / np.sqrt(len(xx))
ax[1].errorbar(1, u, yerr=yerr, marker="o", 
            capsize=2, lw=1, markerfacecolor="grey", markeredgecolor="k", color="k")    
# ref / ref
xx = np.random.normal(2, 0.03, len(df[(df.area=="PEG") & (df["class"]=="ref_ref")].site.unique()))
ax[1].scatter(
    xx,
    df[(df["class"]=="ref_ref") & (df.area=="PEG")].groupby(by=["site", "area"]).mean()[delta_metric],
    s=s, c="blue"
)
u = df[(df["class"]=="ref_ref") & (df.area=="PEG")].groupby(by=["site", "area"]).mean()[delta_metric].mean()
yerr = df[(df["class"]=="ref_ref") & (df.area=="PEG")].groupby(by=["site", "area"]).mean()[delta_metric].std() / np.sqrt(len(xx))
ax[1].errorbar(2, u, yerr=yerr, marker="o", 
            capsize=2, lw=1, markerfacecolor="blue", markeredgecolor="k", color="k")   
for site in df[df.area=="PEG"].site.unique():
    x1 = df[(df.site==site) & (df["class"]=="tar_tar")][delta_metric].mean()
    x2 = df[(df.site==site) & (df["class"]=="tar_cat")][delta_metric].mean()
    x3 = df[(df.site==site) & (df["class"]=="ref_ref")][delta_metric].mean()
    ax[1].plot([0.2, 0.8], [x1, x2], color="k", lw=0.5)
    ax[1].plot([1.2, 1.8], [x2, x3], color="k", lw=0.5)

ax[1].set_title("PEG")

for a in ax:
    a.axhline(0, linestyle="--", color="k")
    a.set_xticks([])
    a.set_ylabel(r"$\Delta d'$")

f.tight_layout()

## ====== Supplementals ========

## mag of delta for tar / cat is selectively bigger