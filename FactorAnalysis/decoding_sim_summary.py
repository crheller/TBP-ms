"""
Decoding performance for different categories using FA simulations
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
# df_drop = df[["dp_x", "dp_y", "e1", "e2", "site"]].drop_duplicates()
# df = df.loc[df_drop.index]
df["delta"] = (df["dp_y"] - df["dp_x"]) / (df["dp_y"] + df["dp_x"])
df["delta_raw"] = df["dp_y"] - df["dp_x"]

# Look at delta dprime between simulation and raw data
delta_metric = "delta"
s = 25
alpha = 0.4
f, ax = plt.subplots(1, 3, figsize=(9, 3))

ax[0].scatter(
    df[(df.sim=="raw") & (df["class"]=="tar_cat") & (df["area"]=="A1")][delta_metric],
    df[(df.sim==fa1) & (df["class"]=="tar_cat") & (df["area"]=="A1")][delta_metric],
    s=s, alpha=alpha
)
ax[0].scatter(
    df[(df.sim=="raw") & (df["class"]=="tar_tar") & (df["area"]=="A1")][delta_metric],
    df[(df.sim==fa1) & (df["class"]=="tar_tar") & (df["area"]=="A1")][delta_metric],
    s=s, alpha=alpha
)
ax[0].set_title("Gain only")

ax[1].scatter(
    df[(df.sim=="raw") & (df["class"]=="tar_cat") & (df["area"]=="A1")][delta_metric],
    df[(df.sim==fa2) & (df["class"]=="tar_cat") & (df["area"]=="A1")][delta_metric],
    s=s, alpha=alpha
)
ax[1].scatter(
    df[(df.sim=="raw") & (df["class"]=="tar_tar") & (df["area"]=="A1")][delta_metric],
    df[(df.sim==fa2) & (df["class"]=="tar_tar") & (df["area"]=="A1")][delta_metric],
    s=s, alpha=alpha
)
ax[1].set_title("Indep. noise")

ax[2].scatter(
    df[(df.sim=="raw") & (df["class"]=="tar_cat") & (df["area"]=="A1")][delta_metric],
    df[(df.sim==fa3) & (df["class"]=="tar_cat") & (df["area"]=="A1")][delta_metric],
    s=s, alpha=alpha
)
ax[2].scatter(
    df[(df.sim=="raw") & (df["class"]=="tar_tar") & (df["area"]=="A1")][delta_metric],
    df[(df.sim==fa3) & (df["class"]=="tar_tar") & (df["area"]=="A1")][delta_metric],
    s=s, alpha=alpha
)
ax[2].set_title("Full model")

for a in ax:
    mm = np.min(a.get_xlim()+a.get_ylim())
    m = np.max(a.get_xlim()+a.get_ylim())
    a.plot([mm, m], [mm, m], "k--")
    a.set_xlabel("Raw")
    a.set_ylabel("Simulated")
f.suptitle("A1")
f.tight_layout()

f, ax = plt.subplots(1, 3, figsize=(9, 3))

ax[0].scatter(
    df[(df.sim=="raw") & (df["class"]=="tar_cat") & (df["area"]=="PEG")][delta_metric],
    df[(df.sim==fa1) & (df["class"]=="tar_cat") & (df["area"]=="PEG")][delta_metric],
    s=s, alpha=alpha
)
ax[0].scatter(
    df[(df.sim=="raw") & (df["class"]=="tar_tar") & (df["area"]=="PEG")][delta_metric],
    df[(df.sim==fa1) & (df["class"]=="tar_tar") & (df["area"]=="PEG")][delta_metric],
    s=s, alpha=alpha
)
ax[0].set_title("Gain only")

ax[1].scatter(
    df[(df.sim=="raw") & (df["class"]=="tar_cat") & (df["area"]=="PEG")][delta_metric],
    df[(df.sim==fa2) & (df["class"]=="tar_cat") & (df["area"]=="PEG")][delta_metric],
    s=s, alpha=alpha
)
ax[1].scatter(
    df[(df.sim=="raw") & (df["class"]=="tar_tar") & (df["area"]=="PEG")][delta_metric],
    df[(df.sim==fa2) & (df["class"]=="tar_tar") & (df["area"]=="PEG")][delta_metric],
    s=s, alpha=alpha
)
ax[1].set_title("Indep. noise")

ax[2].scatter(
    df[(df.sim=="raw") & (df["class"]=="tar_cat") & (df["area"]=="PEG")][delta_metric],
    df[(df.sim==fa3) & (df["class"]=="tar_cat") & (df["area"]=="PEG")][delta_metric],
    s=s, alpha=alpha
)
ax[2].scatter(
    df[(df.sim=="raw") & (df["class"]=="tar_tar") & (df["area"]=="PEG")][delta_metric],
    df[(df.sim==fa3) & (df["class"]=="tar_tar") & (df["area"]=="PEG")][delta_metric],
    s=s, alpha=alpha
)
ax[2].set_title("Full model")

for a in ax:
    mm = np.min(a.get_xlim()+a.get_ylim())
    m = np.max(a.get_xlim()+a.get_ylim())
    a.plot([mm, m], [mm, m], "k--")
    a.set_xlabel("Raw")
    a.set_ylabel("Simulated")
f.suptitle("PEG")
f.tight_layout()

# Look at "selectivity" as a function of simulation
# define selectivity as the difference in tar_tar vs. tar_cat delta
tc = df[df["class"]=="tar_cat"].groupby(by=["site", "area", "sim"]).mean()
tt = df[df["class"]=="tar_tar"].groupby(by=["site", "area", "sim"]).mean()
selectivity = tc[delta_metric] - tt[delta_metric]
peg_mask = selectivity.index.get_level_values(1)=="PEG"
a1_mask = selectivity.index.get_level_values(1)=="A1"
s1_mask = selectivity.index.get_level_values(2)==fa1
s2_mask = selectivity.index.get_level_values(2)==fa2
s3_mask = selectivity.index.get_level_values(2)==fa3
r_mask = selectivity.index.get_level_values(2)=="raw"

f, ax = plt.subplots(1, 3, figsize=(9, 3))

ax[0].scatter(
    selectivity[peg_mask & r_mask].values,
    selectivity[peg_mask & s1_mask].values
)
ax[0].set_title("Gain only")

ax[1].scatter(
    selectivity[peg_mask & r_mask].values,
    selectivity[peg_mask & s2_mask].values
)
ax[1].set_title("Indep. noise")

ax[2].scatter(
    selectivity[peg_mask & r_mask].values,
    selectivity[peg_mask & s3_mask].values
)
ax[2].set_title("Full model")

for a in ax:
    mm = np.min(a.get_xlim()+a.get_ylim())
    m = np.max(a.get_xlim()+a.get_ylim())
    a.plot([mm, m], [mm, m], "k--")
    a.set_xlabel("Raw selectivity")
    a.set_ylabel("Simulated selectivity")
f.suptitle("PEG")
f.tight_layout()

f, ax = plt.subplots(1, 3, figsize=(9, 3))

ax[0].scatter(
    selectivity[a1_mask & r_mask].values,
    selectivity[a1_mask & s1_mask].values
)
ax[0].set_title("Gain only")

ax[1].scatter(
    selectivity[a1_mask & r_mask].values,
    selectivity[a1_mask & s2_mask].values
)
ax[1].set_title("Indep. noise")

ax[2].scatter(
    selectivity[a1_mask & r_mask].values,
    selectivity[a1_mask & s3_mask].values
)
ax[2].set_title("Full model")

for a in ax:
    mm = np.min(a.get_xlim()+a.get_ylim())
    m = np.max(a.get_xlim()+a.get_ylim())
    a.plot([mm, m], [mm, m], "k--")
    a.set_xlabel("Raw selectivity")
    a.set_ylabel("Simulated selectivity")
f.suptitle("A1")
f.tight_layout()