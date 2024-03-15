"""
Start decoding from first bin of the trial
Presumably, performance should be at or near zero.
If not, we have an overfitting problem.
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
import scipy.ndimage as sd

catch_model = "tbpChoiceDecoding_fs100_decision.cr.ich_DRops.dim2.ddr"
target_model = "tbpChoiceDecoding_fs100_decision.h.m_DRops.dim2.ddr"

twindows = [
    "ws0.0_we0.1_trial_fromfirst", 
    "ws0.1_we0.2_trial_fromfirst", 
    "ws0.2_we0.3_trial_fromfirst", 
    "ws0.3_we0.4_trial_fromfirst", 
    "ws0.4_we0.5_trial_fromfirst"
]

batch = 324
sqrt = True
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])
sites = [s for s in sites if s not in BAD_SITES]


# CATCH ANALYSIS
a1_catch = []
peg_catch = []
a1_catch_pc = []
peg_catch_pc = []
for site in sites:
    try:
        dprimes = []
        pc = []
        for twin in twindows:
            model = catch_model.replace("Decoding_fs100_", f"Decoding_fs100_{twin}_")
            res = pd.read_pickle(results_file(RESULTS_DIR, site, batch, model, "output.pickle"))    
            res["site"] = site
            area = nd.pd_query(sql=f"SELECT area from sCellFile where cellid like '%{site}%'")
            area = area.iloc[0][0]
            res["area"] = area
            if sqrt:
                res["dp"] = np.sqrt(res["dp"])
            dprimes.append(res["dp"].values[0])
            pc.append(res["percent_correct"].values[0])

        if area=="A1":
            a1_catch.append(dprimes) # / np.max(dprimes))
            a1_catch_pc.append(pc) # / np.max(pc))
        else:
            peg_catch.append(dprimes) # / np.max(dprimes))
            peg_catch_pc.append(pc) # / np.max(pc))
    except:
        print(f"model didn't exsit for site {site}. Prob too few reps")

a1_catch = np.stack(a1_catch_pc)
peg_catch = np.stack(peg_catch_pc)
a1_catch_pc = np.stack(a1_catch_pc)
peg_catch_pc = np.stack(peg_catch_pc)

t = np.linspace(-0.5, 0.5, len(twindows))
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


# TARGET ANALYSIS
a1_target_inf = []
peg_target_inf = []
a1_target_0 = []
peg_target_0 = []
a1_target_5 = []
peg_target_5 = []
a1_target_pc_inf = []
peg_target_pc_inf = []
a1_target_pc_0 = []
peg_target_pc_0 = []
a1_target_pc_5 = []
peg_target_pc_5 = []
for site in sites:
    try:
        dprimes_inf = []
        dprimes_0 = []
        dprimes_5 = []
        pc_inf = []
        pc_0 = []
        pc_5 = []
        for twin in twindows:
            model = target_model.replace("Decoding_fs100_", f"Decoding_fs100_{twin}_")
            res = pd.read_pickle(results_file(RESULTS_DIR, site, batch, model, "output.pickle"))    
            res["site"] = site
            area = nd.pd_query(sql=f"SELECT area from sCellFile where cellid like '%{site}%'")
            area = area.iloc[0][0]
            res["area"] = area
            if sqrt:
                res["dp"] = np.sqrt(res["dp"])
            
            dbmask_inf =  res["stimulus"].str.contains("InfdB").values
            if sum(dbmask_inf)==1:
                dprimes_inf.append(res["dp"].values[dbmask_inf][0])
                pc_inf.append(res["percent_correct"].values[dbmask_inf][0])
            else:
                dprimes_inf.append(np.nan)
                pc_inf.append(np.nan)

            dbmask_0 =  res["stimulus"].str.contains("\+0dB").values
            if sum(dbmask_0)==1:
                dprimes_0.append(res["dp"].values[dbmask_0][0])
                pc_0.append(res["percent_correct"].values[dbmask_0][0])
            else:
                dprimes_0.append(np.nan)
                pc_0.append(np.nan)

            dbmask_5 =  res["stimulus"].str.contains("\-5dB").values
            if sum(dbmask_5)==1:
                dprimes_5.append(res["dp"].values[dbmask_5][0])
                pc_5.append(res["percent_correct"].values[dbmask_5][0])
            else:
                dprimes_5.append(np.nan)
                pc_5.append(np.nan)

        if area=="A1":
            a1_target_inf.append(dprimes_inf) # / np.max(dprimes))
            a1_target_0.append(dprimes_0)
            a1_target_5.append(dprimes_5)

            a1_target_pc_inf.append(pc_inf)
            a1_target_pc_0.append(pc_0)
            a1_target_pc_5.append(pc_5)
        else:
            peg_target_inf.append(dprimes_inf) # / np.max(dprimes))
            peg_target_0.append(dprimes_0)
            peg_target_5.append(dprimes_5)

            peg_target_pc_inf.append(pc_inf)
            peg_target_pc_0.append(pc_0)
            peg_target_pc_5.append(pc_5)
    except:
        print(f"model didn't exsit for site {site}. Prob too few reps")

a1_target_inf = np.stack(a1_target_inf)
peg_target_inf = np.stack(peg_target_inf)
a1_target_0 = np.stack(a1_target_0)
peg_target_0 = np.stack(peg_target_0)
a1_target_5 = np.stack(a1_target_5)
peg_target_5 = np.stack(peg_target_5)

a1_target_pc_inf = np.stack(a1_target_pc_inf)
peg_target_pc_inf = np.stack(peg_target_pc_inf)
a1_target_pc_0 = np.stack(a1_target_pc_0)
peg_target_pc_0 = np.stack(peg_target_pc_0)
a1_target_pc_5 = np.stack(a1_target_pc_5)
peg_target_pc_5 = np.stack(peg_target_pc_5)



# line plot of choice for each snr level / recording site
sigma = 1
a1t_inf_smooth = sd.gaussian_filter1d(a1_target_pc_inf, sigma, axis=1)
a1t_0_smooth = sd.gaussian_filter1d(a1_target_pc_0, sigma, axis=1)
a1t_5_smooth = sd.gaussian_filter1d(a1_target_pc_5, sigma, axis=1)
pegt_inf_smooth = sd.gaussian_filter1d(peg_target_pc_inf, sigma, axis=1)
pegt_0_smooth = sd.gaussian_filter1d(peg_target_pc_0, sigma, axis=1)
pegt_5_smooth = sd.gaussian_filter1d(peg_target_pc_5, sigma, axis=1)

f, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for (j, i) in enumerate(np.argmax(a1t_inf_smooth, axis=1)):
    if np.isfinite(a1t_inf_smooth[j, 0]):
        ax[0].plot(a1t_inf_smooth[j, :], color="grey")
        ax[0].plot(i, a1t_inf_smooth[j, i], "*", color="k")
ax[0].set_title("Inf dB")

for (j, i) in enumerate(np.argmax(a1t_0_smooth, axis=1)):
    if np.isfinite(a1t_0_smooth[j, 0]):
        ax[1].plot(a1t_0_smooth[j, :], color="grey")
        ax[1].plot(i, a1t_0_smooth[j, i], "*", color="k")
ax[1].set_title("0 dB")

for (j, i) in enumerate(np.argmax(a1t_5_smooth, axis=1)):
    if np.isfinite(a1t_5_smooth[j, 0]):
        ax[2].plot(a1t_5_smooth[j, :], color="grey")
        ax[2].plot(i, a1t_5_smooth[j, i], "*", color="k")
ax[2].set_title("-5 dB")
for a in ax:
    a.set_xlabel("Time (sec)")
    a.set_ylabel("choice prob.")
ax[2].legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")

f.suptitle("A1")
f.tight_layout()

# PEG
f, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for (j, i) in enumerate(np.argmax(pegt_inf_smooth, axis=1)):
    if np.isfinite(pegt_inf_smooth[j, 0]):
        ax[0].plot(pegt_inf_smooth[j, :], color="grey")
        ax[0].plot(i, pegt_inf_smooth[j, i], "*", color="k")
ax[0].set_title("Inf dB")

for (j, i) in enumerate(np.argmax(pegt_0_smooth, axis=1)):
    if np.isfinite(pegt_0_smooth[j, 0]):
        ax[1].plot(pegt_0_smooth[j, :], color="grey")
        ax[1].plot(i, pegt_0_smooth[j, i], "*", color="k")
ax[1].set_title("0 dB")

for (j, i) in enumerate(np.argmax(pegt_5_smooth, axis=1)):
    if np.isfinite(pegt_5_smooth[j, 0]):
        ax[2].plot(pegt_5_smooth[j, :], color="grey")
        ax[2].plot(i, pegt_5_smooth[j, i], "*", color="k")
ax[2].set_title("-5 dB")
for a in ax:
    a.set_xlabel("Time (sec)")
    a.set_ylabel("choice prob.")
ax[2].legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")

f.suptitle("PEG")
f.tight_layout()
