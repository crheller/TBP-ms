"""
Look at decoding across time to answer question:
    Does choice information emerge in one area before the other?
"""
import matplotlib.pyplot as plt
import pandas as pd
import nems_lbhb.tin_helpers as thelp
import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from path_helpers import results_file
from settings import RESULTS_DIR, BAD_SITES
import numpy as np
import nems0.db as nd
import scipy.stats as ss

catch_model = "tbpChoiceDecoding_fs100_decision.cr.ich_DRops.dim2.ddr"
target_model = "tbpChoiceDecoding_fs100_decision.h.m_DRops.dim2.ddr"
twindows = [
    "ws0.0_we0.05", 
    "ws0.0_we0.1", 
    "ws0.0_we0.15", 
    "ws0.0_we0.2", 
    "ws0.0_we0.25", 
    "ws0.0_we0.3", 
    "ws0.0_we0.35", 
    "ws0.0_we0.4", 
    "ws0.0_we0.45",
    "ws0.0_we0.5"
]

batch = 324
sqrt = True
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])
sites = [s for s in sites if s not in BAD_SITES]

a1_catch = []
peg_catch = []
for site in sites:
    try:
        dprimes = []
        for twin in twindows:
            model = catch_model.replace("Decoding_fs100_", f"Decoding_fs100_{twin}_")
            res = pd.read_pickle(results_file(RESULTS_DIR, site, batch, model, "output.pickle"))    
            res["site"] = site
            area = nd.pd_query(sql="SELECT area from sCellFile where cellid like %s", params=(f"%{site}%",))
            area = area.iloc[0][0]
            res["area"] = area
            if sqrt:
                res["dp"] = np.sqrt(res["dp"])
            dprimes.append(res["dp"].values[0])
        if area=="A1":
            a1_catch.append(dprimes) #/ np.max(dprimes))
        else:
            peg_catch.append(dprimes) #/ np.max(dprimes))
    except:
        print(f"model didn't exsit for site {site}. Prob too few reps")

a1_catch = np.stack(a1_catch)
peg_catch = np.stack(peg_catch)

t = np.linspace(-0.1, 0.4, len(twindows))
f, ax = plt.subplots(1, 1, figsize=(5, 3))

u = a1_catch.mean(axis=0)
sem = a1_catch.std(axis=0) / np.sqrt(a1_catch.shape[0])
ax.plot(t, u, label="A1")
ax.fill_between(t, u-sem, u+sem, alpha=0.3, lw=0)

u = peg_catch.mean(axis=0)
sem = peg_catch.std(axis=0) / np.sqrt(peg_catch.shape[0])
ax.plot(t, u, label="PEG")
ax.fill_between(t, u-sem, u+sem, alpha=0.3, lw=0)

ax.axvline(0, linestyle="--", color="k")
ax.axvline(0.3, linestyle="--", color="k")

ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
ax.set_xlabel("Time from sound onset (s)")
ax.set_ylabel("Choice decoding (d-prime)")

f, ax = plt.subplots(1, 2, figsize=(3, 6))

for i in range(a1_catch.shape[0]):
    ax[0].plot([0, 1], [a1_catch[i, 3], a1_catch[i, -1]], "o-", color="tab:blue", alpha=0.5)
ax[0].plot([0, 1], [np.nanmedian(a1_catch[:, 3], axis=0), np.nanmedian(a1_catch[:, -1], axis=0)], 
                        "o-", lw=2, markersize=10, color="tab:blue")

for i in range(peg_catch.shape[0]):
    ax[1].plot([0, 1], [peg_catch[i, 3], peg_catch[i, -1]], "o-", color="tab:orange", alpha=0.5)
ax[1].plot([0, 1], [np.nanmedian(peg_catch[:, 3], axis=0), np.nanmedian(peg_catch[:, -1], axis=0)], 
                        "o-", lw=2, markersize=10, color="tab:orange")

for a in ax:
    a.set_ylabel("d-prime")
    a.set_xlabel("T window")
    a.set_xlim([-0.5, 1.5])
    a.set_xticks([0, 1])
    a.set_xticklabels(["onset", "offset"])

f.tight_layout()



a1_target = []
peg_target = []
for site in sites:
    try:
        dprimes = []
        for twin in twindows:
            model = target_model.replace("Decoding_fs100_", f"Decoding_fs100_{twin}_")
            res = pd.read_pickle(results_file(RESULTS_DIR, site, batch, model, "output.pickle"))    
            res["site"] = site
            area = nd.pd_query(sql="SELECT area from sCellFile where cellid like %s", params=(f"%{site}%",))
            area = area.iloc[0][0]
            res["area"] = area
            if sqrt:
                res["dp"] = np.sqrt(res["dp"])
            
            dbmask = res["stimulus"].str.contains("-5dB").values
            if sum(dbmask)==1:
                dprimes.append(res["dp"].values[dbmask][0])
            else:
                dprimes.append(np.nan)
        if area=="A1":
            a1_target.append(dprimes) # / np.max(dprimes))
        else:
            peg_target.append(dprimes) # / np.max(dprimes))
    except:
        print(f"model didn't exsit for site {site}. Prob too few reps")

a1_target = np.stack(a1_target)
peg_target = np.stack(peg_target)

t = np.linspace(-0.1, 0.4, len(twindows))
f, ax = plt.subplots(1, 1, figsize=(5, 3))

u = np.nanmean(a1_target, axis=0)
sem = np.nanstd(a1_target, axis=0) / np.sqrt(np.sum(np.isnan(a1_target)==False, axis=0))
ax.plot(t, u, label="A1")
ax.fill_between(t, u-sem, u+sem, alpha=0.3, lw=0)

u = np.nanmean(peg_target, axis=0)
sem = np.nanstd(peg_target, axis=0) / np.sqrt(np.sum(np.isnan(peg_target)==False, axis=0))
ax.plot(t, u, label="PEG")
ax.fill_between(t, u-sem, u+sem, alpha=0.3, lw=0)

ax.axvline(0, linestyle="--", color="k")
ax.axvline(0.3, linestyle="--", color="k")

ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
ax.set_xlabel("Time from sound onset (s)")
ax.set_ylabel("Choice decoding (d-prime)")

# compare beginning of sound to final timepoint
f, ax = plt.subplots(1, 2, figsize=(3, 6))

for i in range(a1_target.shape[0]):
    ax[0].plot([0, 1], [a1_target[i, 3], a1_target[i, -1]], "o-", color="tab:blue", alpha=0.5)
ax[0].plot([0, 1], [np.nanmedian(a1_target[:, 3], axis=0), np.nanmedian(a1_target[:, -1], axis=0)], 
                        "o-", lw=2, markersize=10, color="tab:blue")

for i in range(peg_target.shape[0]):
    ax[1].plot([0, 1], [peg_target[i, 3], peg_target[i, -1]], "o-", color="tab:orange", alpha=0.5)
ax[1].plot([0, 1], [np.nanmedian(peg_target[:, 3], axis=0), np.nanmedian(peg_target[:, -1], axis=0)], 
                        "o-", lw=2, markersize=10, color="tab:orange")

for a in ax:
    a.set_ylabel("d-prime")
    a.set_xlabel("T window")
    a.set_xlim([-0.5, 1.5])
    a.set_xticks([0, 1])
    a.set_xticklabels(["onset", "offset"])

f.tight_layout()
