import charlieTools.TBP_ms.loaders as loaders
import charlieTools.plotting as cplt
import nems_lbhb.tin_helpers as thelp
import nems0.db as nd
import scipy.stats as ss
from itertools import combinations
import os
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

figpath = "/auto/users/hellerc/code/projects/TBP-ms/figure_files/fig5"

batch = 324
sqrt = True
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])

factor_models = ["_FAperstim.0", "_FAperstim.1", "_FAperstim.3", "_FAperstim.4", ""]
amodel = 'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'
pmodel = 'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'
sites = [s for s in sites if s not in BAD_SITES]
active = []
passive = []
for site in sites:
    for fa in factor_models:
        try:
            ares = pd.read_pickle(results_file(RESULTS_DIR, site, batch, amodel+fa, "output.pickle"))
            pres = pd.read_pickle(results_file(RESULTS_DIR, site, batch, pmodel+fa, "output.pickle"))
            ares["site"] = site; pres["site"] = site
            area = nd.pd_query(sql="SELECT area from sCellFile where cellid like %s", params=(f"%{site}%",))
            area = area.iloc[0][0]
            ares["area"] = area; pres["area"] = area
            ares["FA"] = fa.split(".")[-1]; pres["FA"] = fa.split(".")[-1]; 
            if sqrt:
                ares["dp"] = np.sqrt(ares["dp"])
                pres["dp"] = np.sqrt(pres["dp"])
            active.append(ares)
            passive.append(pres)
        except:
            print(f"results not found for site: {site}")
    
active = pd.concat(active)
passive = pd.concat(passive)
active = active[active["class"].isin(["tar_cat", "tar_tar"])]
passive = passive[passive["class"].isin(["tar_cat", "tar_tar"])]
df = passive.merge(active, on=["site", "FA", "class", "e1", "e2", "area"])
df_drop = df[["dp_x", "dp_y", "e1", "e2", "site", "FA"]].drop_duplicates()
df = df.loc[df_drop.index]
df["delta"] = (df["dp_y"] - df["dp_x"]) / (df["dp_y"] + df["dp_x"])
df["delta_raw"] = df["dp_y"] - df["dp_x"]


# overall delta dprime for tar vs. catch
print("DELTA DPRIME")
f, ax = plt.subplots(1, 1, figsize=(4, 2))
f2, ax2 = plt.subplots(1, 1, figsize=(1, 2))
for col, area in zip(["tab:blue", "tab:orange"], ["A1", "PEG"]):
    u = []
    se = []
    uall = []
    for i, fa in enumerate(factor_models):
        # get mean tar vs. catch delta dprime for each site
        mask = (df.area==area) & (df.FA==fa.split(".")[-1])
        tc_mean_dprime = df[mask & (df["class"]=="tar_cat")].groupby(by="site").mean()["delta"]
        # get mean tar vs. tar delta dprime for each site
        tt_mean_dprime = df[mask & (df["class"]=="tar_tar")].groupby(by="site").mean()["delta"]
        uall.append(tc_mean_dprime.values)
        u.append((tc_mean_dprime).mean())
        se.append((tc_mean_dprime).std() / np.sqrt(tc_mean_dprime.shape[0]))
    ax.plot(range(len(u)-1), u[:-1], color=col)
    ax.scatter(len(u)-1, u[-1], color=col)
    
    # compute stepwise p-values
    cc = []
    for i in range(len(u)-1):
        _cc, pval = ss.pearsonr(uall[i], uall[-1])
        cc.append(_cc)
        print(f"{area}, pvalue of correlation for {i} vs. raw data: {pval}")
    ax2.plot(cc, color=col, linestyle="-")
ax2.set_ylim((None, 1))
for a in [ax, ax2]:
    a.axhline(0, linestyle="--", color="k")
f.savefig(os.path.join(figpath, "delta_dprime_vals.svg"), dpi=500)
f2.savefig(os.path.join(figpath, "delta_dprime_cc.svg"), dpi=500)

# selectivity
print("SELECTIVITY")
f, ax = plt.subplots(1, 1, figsize=(4, 2))
f2, ax2 = plt.subplots(1, 1, figsize=(1, 2))
for col, area in zip(["tab:blue", "tab:orange"], ["A1", "PEG"]):
    u = []
    se = []
    uall = []
    for i, fa in enumerate(factor_models):
        # get mean tar vs. catch delta dprime for each site
        mask = (df.area==area) & (df.FA==fa.split(".")[-1])
        tc_mean_dprime = df[mask & (df["class"]=="tar_cat")].groupby(by="site").mean()["delta"]
        # get mean tar vs. tar delta dprime for each site
        tt_mean_dprime = df[mask & (df["class"]=="tar_tar")].groupby(by="site").mean()["delta"]
        uall.append((tc_mean_dprime - tt_mean_dprime).values)
        u.append((tc_mean_dprime - tt_mean_dprime).mean())
        se.append((tc_mean_dprime - tt_mean_dprime).std() / np.sqrt(tc_mean_dprime.shape[0]))
    ax.plot(range(len(u)-1), u[:-1], color=col)
    ax.scatter(len(u)-1, u[-1], color=col)
    
    # compute stepwise p-values
    cc = []
    for i in range(len(u)-1):
        _cc, pval = ss.pearsonr(uall[i], uall[-1])
        cc.append(_cc)
        print(f"{area}, pvalue of correlation for {i} vs. raw data: {pval}")
    ax2.plot(cc, color=col, linestyle="-")
ax2.set_ylim((None, 1))
for a in [ax, ax2]:
    a.axhline(0, linestyle="--", color="k")
f.savefig(os.path.join(figpath, "selectivity_vals.svg"), dpi=500)
f2.savefig(os.path.join(figpath, "selectivity_cc.svg"), dpi=500)