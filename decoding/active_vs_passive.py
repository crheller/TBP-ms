"""
Simple script to load active vs. passive decding,
sort them based on category, and plot per site
"""
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from path_helpers import results_file
from settings import RESULTS_DIR
import numpy as np
import nems.db as nd

batch = 324
sqrt = True
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])
amodel = 'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'
pmodel = 'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'

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
    except:
        print(f"results not found for site: {site}")

    active.append(ares)
    passive.append(pres)
active = pd.concat(active)
passive = pd.concat(passive)

df = passive.merge(active, on=["site", "class", "e1", "e2", "area"])
df_drop = df[["dp_x", "dp_y", "e1", "e2", "site"]].drop_duplicates()
df = df.loc[df_drop.index]

# plot summary per site
f, ax = plt.subplots(2, 4, figsize=(20, 10))

# A1
cm = (df["class"] == "tar_tar") & (df["area"]=="A1")
g = df[["dp_x", "dp_y", "class", "area", "site"]][cm].groupby(by="site").mean()
ax[0, 0].scatter(g["dp_x"], g["dp_y"], color="red", edgecolor="k", s=50)
mm = max(ax[0, 0].get_xlim()+ax[0, 0].get_ylim())
ax[0, 0].plot([0, mm], [0, mm], "k--")
ax[0, 0].set_title("TAR vs. TAR, A1")
tt_diff = g["dp_y"] - g["dp_x"]

cm = (df["class"] == "ref_ref") & (df["area"]=="A1")
g = df[["dp_x", "dp_y", "class", "area", "site"]][cm].groupby(by="site").mean()
ax[0, 1].scatter(g["dp_x"], g["dp_y"], color="blue", edgecolor="k", s=50)
mm = max(ax[0, 1].get_xlim()+ax[0, 1].get_ylim())
ax[0, 1].plot([0, mm], [0, mm], "k--")
ax[0, 1].set_title("REF vs. REF, A1")
rr_diff = g["dp_y"] - g["dp_x"]

cm = (df["class"] == "tar_cat") & (df["area"]=="A1")
g = df[["dp_x", "dp_y", "class", "area", "site"]][cm].groupby(by="site").mean()
ax[0, 2].scatter(g["dp_x"], g["dp_y"], color="grey", edgecolor="k", s=50)
mm = max(ax[0, 2].get_xlim()+ax[0, 2].get_ylim())
ax[0, 2].plot([0, mm], [0, mm], "k--")
ax[0, 2].set_title("TAR vs. CAT, A1")
tc_diff = g["dp_y"] - g["dp_x"]

ri = np.random.normal(0, 0.1, len(tt_diff))
ax[0, 3].scatter([0], tt_diff.median(), color="red", edgecolor="k", s=75)
ax[0, 3].scatter(ri, tt_diff, alpha=0.5, s=25, color="red")
ri = np.random.normal(1, 0.1, len(rr_diff))
ax[0, 3].scatter([1], rr_diff.median(), color="blue", edgecolor="k", s=75)
ax[0, 3].scatter(ri, rr_diff, alpha=0.5, s=25, color="blue")
ri = np.random.normal(2, 0.1, len(tc_diff))
ax[0, 3].scatter([2], tc_diff.median(), color="grey", edgecolor="k", s=75)
ax[0, 3].scatter(ri, tc_diff, alpha=0.5, s=25, color="grey")
ax[0, 3].axhline(0, linestyle="--", color="k")


# PEG
cm = (df["class"] == "tar_tar") & (df["area"]=="PEG")
g = df[["dp_x", "dp_y", "class", "area", "site"]][cm].groupby(by="site").mean()
ax[1, 0].scatter(g["dp_x"], g["dp_y"], color="red", edgecolor="k", s=50)
mm = max(ax[1, 0].get_xlim()+ax[1, 0].get_ylim())
ax[1, 0].plot([0, mm], [0, mm], "k--")
ax[1, 0].set_title("TAR vs. TAR, PEG")
tt_diff = g["dp_y"] - g["dp_x"]

cm = (df["class"] == "ref_ref") & (df["area"]=="PEG")
g = df[["dp_x", "dp_y", "class", "area", "site"]][cm].groupby(by="site").mean()
ax[1, 1].scatter(g["dp_x"], g["dp_y"], color="blue", edgecolor="k", s=50)
mm = max(ax[1, 1].get_xlim()+ax[1, 1].get_ylim())
ax[1, 1].plot([0, mm], [0, mm], "k--")
ax[1, 1].set_title("REF vs. REF, PEG")
rr_diff = g["dp_y"] - g["dp_x"]

cm = (df["class"] == "tar_cat") & (df["area"]=="PEG")
g = df[["dp_x", "dp_y", "class", "area", "site"]][cm].groupby(by="site").mean()
ax[1, 2].scatter(g["dp_x"], g["dp_y"], color="grey", edgecolor="k", s=50)
mm = max(ax[1, 2].get_xlim()+ax[1, 2].get_ylim())
ax[1, 2].plot([0, mm], [0, mm], "k--")
ax[1, 2].set_title("TAR vs. CAT, PEG")
tc_diff = g["dp_y"] - g["dp_x"]


ri = np.random.normal(0, 0.1, len(tt_diff))
ax[1, 3].scatter([0], tt_diff.median(), color="red", edgecolor="k", s=75)
ax[1, 3].scatter(ri, tt_diff, alpha=0.5, s=25, color="red")
ri = np.random.normal(1, 0.1, len(rr_diff))
ax[1, 3].scatter([1], rr_diff.median(), color="blue", edgecolor="k", s=75)
ax[1, 3].scatter(ri, rr_diff, alpha=0.5, s=25, color="blue")
ri = np.random.normal(2, 0.1, len(tc_diff))
ax[1, 3].scatter([2], tc_diff.median(), color="grey", edgecolor="k", s=75)
ax[1, 3].scatter(ri, tc_diff, alpha=0.5, s=25, color="grey")
ax[1, 3].axhline(0, linestyle="--", color="k")

f.tight_layout()