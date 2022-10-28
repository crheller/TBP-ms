import nems0.db as nd

import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
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

sites = [s for s in sites if s not in BAD_SITES]
df = pd.DataFrame(columns=["asv", "psv", "als", "pls", "adim", "pdim", "site", "area"])
for i, site in enumerate(sites):
    d = pd.read_pickle(os.path.join(RESULTS_DIR, "factor_analysis", str(batch), site, "FA.pickle"))
    area = nd.pd_query(sql="SELECT area from sCellFile where cellid like %s", params=(f"%{site}%",)).iloc[0][0]
    df.loc[i, :] = [
        d["active_sv"],
        d["passive_sv"],
        d["active_loading_sim"],
        d["passive_loading_sim"],
        d["active_dim95"],
        d["passive_dim95"],
        site,
        area
    ]

f, ax = plt.subplots(1, 3, figsize=(9, 3))
mask = df.area == "A1"
ax[0].scatter(df[mask]["psv"], df[mask]["asv"], c="k", s=25)
ax[0].set_title("Shared variance")

ax[1].scatter(df[mask]["pls"], df[mask]["als"], c="k", s=25)
ax[1].set_title("Loading similarity")

ax[2].scatter(df[mask]["pdim"], df[mask]["adim"], c="k", s=25)
ax[2].set_title("Dimensionality")

for a in ax:
    mm = np.min(a.get_xlim()+a.get_ylim())
    m = np.max(a.get_xlim()+a.get_ylim())
    a.plot([mm, m], [mm, m], "k--")
    a.set_xlabel("Passive")
    a.set_ylabel("Active")
f.tight_layout()

f, ax = plt.subplots(1, 3, figsize=(9, 3))
mask = df.area == "PEG"
ax[0].scatter(df[mask]["psv"], df[mask]["asv"], c="k", s=25)
ax[0].set_title("Shared variance")

ax[1].scatter(df[mask]["pls"], df[mask]["als"], c="k", s=25)
ax[1].set_title("Loading similarity")

ax[2].scatter(df[mask]["pdim"], df[mask]["adim"], c="k", s=25)
ax[2].set_title("Dimensionality")

for a in ax:
    mm = np.min(a.get_xlim()+a.get_ylim())
    m = np.max(a.get_xlim()+a.get_ylim())
    a.plot([mm, m], [mm, m], "k--")
    a.set_xlabel("Passive")
    a.set_ylabel("Active")
f.tight_layout()