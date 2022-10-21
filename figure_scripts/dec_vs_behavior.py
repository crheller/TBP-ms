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

batch = 324
sqrt = True
sites = [s for s in nd.get_batch_sites(batch)[0] if s not in BAD_SITES]


# load neural dprime
amodel = 'tbpDecoding_mask.h.cr.m_decmask.h.cr.m.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'
pmodel = 'tbpDecoding_mask.pa_decmask.h.cr.m.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'
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

# delta dprime
f, ax = plt.subplots(1, 2, figsize=(8, 4))

r, p = ss.pearsonr(a1_merge["dprime"], a1_merge["delta_raw"])
leg = f"r={round(r, 3)}, p={round(p, 3)}"
ax[0].scatter(a1_merge["dprime"], a1_merge["delta_raw"], 
                s=25, c="k")
ax[0].set_title(f"A1, {leg}")
r, p = ss.pearsonr(peg_merge["dprime"], peg_merge["delta_raw"])
leg = f"r={round(r, 3)}, p={round(p, 3)}"
ax[1].scatter(peg_merge["dprime"], peg_merge["delta_raw"], 
                s=25, c="k")
ax[1].set_title(f"PEG, {leg}")
for a in ax:
    a.set_xlabel(r"Behavioral $d'$")
    a.set_ylabel(r"Neural $\Delta d'$")
    a.axhline(0, linestyle="--", color="grey")
    a.axvline(0, linestyle="--", color="grey")
f.tight_layout()

# abs dprime
f, ax = plt.subplots(1, 2, figsize=(8, 4))

r, p = ss.pearsonr(a1_merge["dprime"], a1_merge["dp_x"])
leg = f"r={round(r, 3)}, p={round(p, 3)}"
ax[0].scatter(a1_merge["dprime"], a1_merge["dp_x"], 
                s=25, c="k")
ax[0].set_title(f"A1, {leg}")
r, p = ss.pearsonr(peg_merge["dprime"], peg_merge["dp_x"])
leg = f"r={round(r, 3)}, p={round(p, 3)}"
ax[1].scatter(peg_merge["dprime"], peg_merge["dp_x"], 
                s=25, c="k")
ax[1].set_title(f"PEG, {leg}")
for a in ax:
    a.set_xlabel(r"Behavioral $d'$")
    a.set_ylabel(r"Neural $d'$")
    a.axhline(0, linestyle="--", color="grey")
    a.axvline(0, linestyle="--", color="grey")
f.tight_layout()