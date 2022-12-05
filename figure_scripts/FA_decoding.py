"""
Summarize delta decoding performance across different FA simulations
Goal is to illustrate that selectivity "pops" out for only the 
full simulation, including modulated correlations.
"""
import nems0.db as nd

import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from path_helpers import results_file
import os
from settings import RESULTS_DIR, BAD_SITES

import pickle
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
amodel = 'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'
pmodel = 'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'
fa1 = "FAperstim.1"
fa2 = "FAperstim.3"
fa3 = "FAperstim.4"
fa_models = ["raw", fa1, fa2, fa3]
sites = [s for s in sites if s not in BAD_SITES]
active = []
passive = []
for site in sites:
    for fa in fa_models:
        try:
            if fa == "raw":
                am = amodel
                pm = pmodel
            else:
                am = amodel+f"_{fa}"
                pm = pmodel+f"_{fa}"
            ares = pd.read_pickle(results_file(RESULTS_DIR, site, batch, am, "output.pickle"))
            pres = pd.read_pickle(results_file(RESULTS_DIR, site, batch, pm, "output.pickle"))
            ares["site"] = site
            pres["site"] = site
            ares["sim"] = fa
            pres["sim"] = fa
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
df = passive.merge(active, on=["site", "class", "e1", "e2", "area", "sim"])
df["delta"] = (df["dp_y"] - df["dp_x"]) / (df["dp_y"] + df["dp_x"])
df["delta_raw"] = df["dp_y"] - df["dp_x"]


# Make figure
s = 10
delta_metric = "delta"
f, ax = plt.subplots(2, 4, figsize=(6, 4), sharey=True)

for i, sim in enumerate(fa_models):
    # A1
    mask = (df.area=="A1") & (df["class"]=="tar_tar") & (df["sim"]==sim)
    xx = np.random.normal(0, 0.03, len(df[mask].site.unique()))
    ax[0, i].scatter(
        xx,
        df[mask].groupby(by=["site", "area"]).mean()[delta_metric],
        s=s, c="r"
    )
    u = df[mask].groupby(by=["site", "area"]).mean()[delta_metric].mean()
    yerr = df[mask].groupby(by=["site", "area"]).mean()[delta_metric].std() / np.sqrt(len(xx))
    ax[0, i].errorbar(0, u, yerr=yerr, marker="o", 
                capsize=2, lw=1, markerfacecolor="r", markeredgecolor="k", color="k")     
    
    mask = (df.area=="A1") & (df["class"]=="tar_cat") & (df["sim"]==sim)
    xx = np.random.normal(1, 0.03, len(df[mask].site.unique()))
    ax[0, i].scatter(
        xx,
        df[mask].groupby(by=["site", "area"]).mean()[delta_metric],
        s=s, c="grey"
    )
    u = df[mask].groupby(by=["site", "area"]).mean()[delta_metric].mean()
    yerr = df[mask].groupby(by=["site", "area"]).mean()[delta_metric].std() / np.sqrt(len(xx))
    ax[0, i].errorbar(1, u, yerr=yerr, marker="o", 
                capsize=2, lw=1, markerfacecolor="grey", markeredgecolor="k", color="k") 

    for site in df[df.area=="A1"].site.unique():
        x1 = df[(df.site==site) & (df["class"]=="tar_tar") & (df["sim"]==sim)][delta_metric].mean()
        x2 = df[(df.site==site) & (df["class"]=="tar_cat") & (df["sim"]==sim)][delta_metric].mean()
        ax[0, i].plot([0.2, 0.8], [x1, x2], color="k", lw=0.5)

    ax[0, i].set_title(f"A1, {sim}")

    # PEG
    mask = (df.area=="PEG") & (df["class"]=="tar_tar") & (df["sim"]==sim)
    xx = np.random.normal(0, 0.03, len(df[mask].site.unique()))
    ax[1, i].scatter(
        xx,
        df[mask].groupby(by=["site", "area"]).mean()[delta_metric],
        s=s, c="r"
    )
    u = df[mask].groupby(by=["site", "area"]).mean()[delta_metric].mean()
    yerr = df[mask].groupby(by=["site", "area"]).mean()[delta_metric].std() / np.sqrt(len(xx))
    ax[1, i].errorbar(0, u, yerr=yerr, marker="o", 
                capsize=2, lw=1, markerfacecolor="r", markeredgecolor="k", color="k")     
    
    mask = (df.area=="PEG") & (df["class"]=="tar_cat") & (df["sim"]==sim)
    xx = np.random.normal(1, 0.03, len(df[mask].site.unique()))
    ax[1, i].scatter(
        xx,
        df[mask].groupby(by=["site", "area"]).mean()[delta_metric],
        s=s, c="grey"
    )
    u = df[mask].groupby(by=["site", "area"]).mean()[delta_metric].mean()
    yerr = df[mask].groupby(by=["site", "area"]).mean()[delta_metric].std() / np.sqrt(len(xx))
    ax[1, i].errorbar(1, u, yerr=yerr, marker="o", 
                capsize=2, lw=1, markerfacecolor="grey", markeredgecolor="k", color="k") 

    for site in df[df.area=="PEG"].site.unique():
        x1 = df[(df.site==site) & (df["class"]=="tar_tar") & (df["sim"]==sim)][delta_metric].mean()
        x2 = df[(df.site==site) & (df["class"]=="tar_cat") & (df["sim"]==sim)][delta_metric].mean()
        ax[1, i].plot([0.2, 0.8], [x1, x2], color="k", lw=0.5)

    ax[1, i].set_title(f"PEG, {sim}")

for a in ax.flatten():
    a.axhline(0, linestyle="--", color="k")
    a.set_xticks([])
f.tight_layout()
