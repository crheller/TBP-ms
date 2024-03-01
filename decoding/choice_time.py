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
import scipy.ndimage as sd

catch_model = "tbpChoiceDecoding_fs100_decision.cr.ich_DRops.dim2.ddr"
target_model = "tbpChoiceDecoding_fs100_decision.h.m_DRops.dim2.ddr"
twindows = [
    "ws0.0_we0.1", 
    "ws0.05_we0.15", 
    "ws0.1_we0.2", 
    "ws0.15_we0.25", 
    "ws0.2_we0.3", 
    "ws0.25_we0.35", 
    "ws0.3_we0.4", 
    "ws0.35_we0.45", 
    "ws0.4_we0.5"
]

batch = 324
sqrt = True
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])
sites = [s for s in sites if s not in BAD_SITES]

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
            area = nd.pd_query(sql="SELECT area from sCellFile where cellid like %s", params=(f"%{site}%",))
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

onw = 2
offw = -1
f, ax = plt.subplots(1, 2, figsize=(3, 6))

for i in range(a1_catch.shape[0]):
    ax[0].plot([0, 1], [a1_catch[i, onw], a1_catch[i, offw]], "o-", color="tab:blue", alpha=0.5)
ax[0].plot([0, 1], [np.nanmedian(a1_catch[:, onw], axis=0), np.nanmedian(a1_catch[:, offw], axis=0)], 
                        "o-", lw=2, markersize=10, color="tab:blue")

for i in range(peg_catch.shape[0]):
    ax[1].plot([0, 1], [peg_catch[i, onw], peg_catch[i, offw]], "o-", color="tab:orange", alpha=0.5)
ax[1].plot([0, 1], [np.nanmedian(peg_catch[:, onw], axis=0), np.nanmedian(peg_catch[:, offw], axis=0)], 
                        "o-", lw=2, markersize=10, color="tab:orange")

for a in ax:
    a.set_ylabel("d-prime")
    a.set_xlabel("T window")
    a.set_xlim([-0.5, 1.5])
    a.set_xticks([0, 1])
    a.set_xticklabels(["onset", "offset"])

f.tight_layout()

a1_p = ss.wilcoxon(a1_catch[:, onw], a1_catch[:, offw]).pvalue
peg_p = ss.wilcoxon(peg_catch[:, onw], peg_catch[:, offw]).pvalue
print(f"A1 catch pval: {a1_p}")
print(f"PEG catch pval: {peg_p}")



# ================= TARGET CHOICE ========================
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
            area = nd.pd_query(sql="SELECT area from sCellFile where cellid like %s", params=(f"%{site}%",))
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



# plot single datasets
f, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(a1_target_inf, vmax=5, vmin=0, cmap="Reds")
for (j, i) in enumerate(np.argmax(a1_target_inf, axis=1)):
    if np.isfinite(a1_target_inf[j, 0]):
        ax[0].plot(i, j, "*", color="k")
ax[0].set_title("Inf dB")

ax[1].imshow(a1_target_0, vmax=5, vmin=0, cmap="Reds")
for (j, i) in enumerate(np.argmax(a1_target_0, axis=1)):
    if np.isfinite(a1_target_0[j, 0]):
        ax[1].plot(i, j, "*", color="k")
ax[1].set_title("0 dB")

ax[2].imshow(a1_target_5, vmax=5, vmin=0, cmap="Reds")
for (j, i) in enumerate(np.argmax(a1_target_5, axis=1)):
    if np.isfinite(a1_target_5[j, 0]):
        ax[2].plot(i, j, "*", color="k")
ax[2].set_title("-5 dB")
for a in ax:
    a.axvline(3.5, color="k", label="lick allowed")
    a.axvline(0.5, color="purple", label="sound on")
    a.set_xticks(np.arange(0, len(twindows), step=2)+0.5)
    a.set_xticklabels(np.round(np.arange(0.1, 0.6, step=0.1), 2))
    a.set_xlabel("Time (sec)")
    a.set_ylabel("Dataset")
ax[2].legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")

f.suptitle("A1")
f.tight_layout()

f, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(peg_target_inf, vmax=5, vmin=0, cmap="Reds")
for (j, i) in enumerate(np.argmax(peg_target_inf, axis=1)):
    if np.isfinite(peg_target_inf[j, 0]):
        ax[0].plot(i, j, "*", color="k")
ax[0].set_title("Inf dB")

ax[1].imshow(peg_target_0, vmax=5, vmin=0, cmap="Reds")
for (j, i) in enumerate(np.argmax(peg_target_0, axis=1)):
    if np.isfinite(peg_target_0[j, 0]):
        ax[1].plot(i, j, "*", color="k")
ax[1].set_title("0 dB")

ax[2].imshow(peg_target_5, vmax=5, vmin=0, cmap="Reds")
for (j, i) in enumerate(np.argmax(peg_target_5, axis=1)):
    if np.isfinite(peg_target_5[j, 0]):
        ax[2].plot(i, j, "*", color="k")
ax[2].set_title("-5 dB")
for a in ax:
    a.axvline(3.5, color="k", label="lick allowed")
    a.axvline(0.5, color="purple", label="sound on")
    a.set_xticks(np.arange(0, len(twindows), step=2)+0.5)
    a.set_xticklabels(np.round(np.arange(0.1, 0.6, step=0.1), 2))
    a.set_xlabel("Time (sec)")
    a.set_ylabel("Dataset")
ax[2].legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")

f.suptitle("PEG")
f.tight_layout()



## LINE PLOT VERSION OF THE ABOVE HEATMAP
# smooth the data?
sigma = 1.2
a1t_inf_smooth = sd.gaussian_filter1d(a1_target_inf, sigma, axis=1)
a1t_0_smooth = sd.gaussian_filter1d(a1_target_0, sigma, axis=1)
a1t_5_smooth = sd.gaussian_filter1d(a1_target_5, sigma, axis=1)
pegt_inf_smooth = sd.gaussian_filter1d(peg_target_inf, sigma, axis=1)
pegt_0_smooth = sd.gaussian_filter1d(peg_target_0, sigma, axis=1)
pegt_5_smooth = sd.gaussian_filter1d(peg_target_5, sigma, axis=1)

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
    a.axvline(3.5, color="k", label="lick allowed")
    a.axvspan(0.5, 4.5, color="purple", alpha=0.3, lw=0, label="sound on")
    a.set_xticks(np.arange(0, len(twindows), step=2)+0.5)
    a.set_xticklabels(np.round(np.arange(0.1, 0.6, step=0.1), 2))
    a.set_xlabel("Time (sec)")
    a.set_ylabel("d-prime")
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
    a.axvline(3.5, color="k", label="lick allowed")
    a.axvspan(0.5, 4.5, color="purple", alpha=0.3, lw=0, label="sound on")
    a.set_xticks(np.arange(0, len(twindows), step=2)+0.5)
    a.set_xticklabels(np.round(np.arange(0.1, 0.6, step=0.1), 2))
    a.set_xlabel("Time (sec)")
    a.set_ylabel("d-prime")
ax[2].legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")

f.suptitle("PEG")
f.tight_layout()


# summary version of the line plots
a1_cmap = plt.get_cmap("Blues", 5)
peg_cmap = plt.get_cmap("Oranges", 5)

f, ax = plt.subplots(2, 2, figsize=(6, 5), sharex=True)

amax = []
ax[0, 0].set_title("A1")
for (i, (a1_data, snr)) in enumerate(zip([a1t_inf_smooth, a1t_0_smooth, a1t_5_smooth], ["inf", "0dB", "-5dB"])):
    for j in range(a1_data.shape[0]):
        if np.isfinite(a1_data[j, 0]):
            ax[0, 0].plot(a1_data[j, :], color=a1_cmap(i+2), lw=0.5, zorder=-1)
    ax[0, 0].plot(np.nanmean(a1_data, axis=0), color=a1_cmap(i+2), lw=2, zorder=0, label=snr)
    amax += list(np.argmax(a1_data, axis=1)[np.isfinite(a1_data[:, 0])])
ax[1, 0].hist(amax, color="royalblue", edgecolor="k", 
                histtype="stepfilled", zorder=0, bins=np.arange(0, 10, step=1), align="left")

pmax = []
ax[0, 1].set_title("PEG")
for (i, (peg_data, snr)) in enumerate(zip([pegt_inf_smooth, pegt_0_smooth, pegt_5_smooth], ["inf", "0dB", "-5dB"])):
    for j in range(peg_data.shape[0]):
        if np.isfinite(peg_data[j, 0]):
            ax[0, 1].plot(peg_data[j, :], color=peg_cmap(i+2), lw=0.5, zorder=-1)
    ax[0, 1].plot(np.nanmean(peg_data, axis=0), color=peg_cmap(i+2), lw=2, zorder=0, label=snr)
    pmax += list(np.argmax(peg_data, axis=1)[np.isfinite(peg_data[:, 0])])
ax[1, 1].hist(pmax, color="tab:orange", edgecolor="k", 
                histtype="stepfilled", zorder=0, bins=np.arange(0, 10, step=1), align="left")

for a in ax.flatten():
    a.axvline(3.5, color="k")
    a.axvspan(0.5, 4.5, color="grey", alpha=0.3, lw=0, zorder=-1)
    a.set_xticks(np.arange(0, len(twindows), step=2)+0.5)
    a.set_xticklabels(np.round(np.arange(0.1, 0.6, step=0.1), 2))

for a in [ax[0, 0], ax[0, 1]]:
    a.set_ylabel("d-prime")
    ax[0, 0].legend(frameon=False, bbox_to_anchor=(0, 1), loc="lower left")
    ax[0, 1].legend(frameon=False, bbox_to_anchor=(0, 1), loc="lower left")
    a.set_xlabel("Time (s)")

for a in [ax[1, 0], ax[1, 1]]:
    a.set_ylabel("count")
    a.set_xlabel("Time of peak d-prime (s)")


f.tight_layout()



# summary version of the line plots using "choice prob."
sigma = 1.2
a1t_inf_smooth_pc = sd.gaussian_filter1d(a1_target_pc_inf, sigma, axis=1)
a1t_0_smooth_pc = sd.gaussian_filter1d(a1_target_pc_0, sigma, axis=1)
a1t_5_smooth_pc = sd.gaussian_filter1d(a1_target_pc_5, sigma, axis=1)
pegt_inf_smooth_pc = sd.gaussian_filter1d(peg_target_pc_inf, sigma, axis=1)
pegt_0_smooth_pc = sd.gaussian_filter1d(peg_target_pc_0, sigma, axis=1)
pegt_5_smooth_pc = sd.gaussian_filter1d(peg_target_pc_5, sigma, axis=1)

a1_cmap = plt.get_cmap("Blues", 5)
peg_cmap = plt.get_cmap("Oranges", 5)

f, ax = plt.subplots(2, 2, figsize=(6, 5), sharex=True)

amax = []
ax[0, 0].set_title("A1")
for (i, (a1_data, snr)) in enumerate(zip([a1t_inf_smooth_pc, a1t_0_smooth_pc, a1t_5_smooth_pc], ["inf", "0dB", "-5dB"])):
    for j in range(a1_data.shape[0]):
        if np.isfinite(a1_data[j, 0]):
            ax[0, 0].plot(a1_data[j, :], color=a1_cmap(i+2), lw=0.5, zorder=-1)
    ax[0, 0].plot(np.nanmean(a1_data, axis=0), color=a1_cmap(i+2), lw=2, zorder=0, label=snr)
    amax += list(np.argmax(a1_data, axis=1)[np.isfinite(a1_data[:, 0])])
ax[1, 0].hist(amax, color="royalblue", edgecolor="k", 
                histtype="stepfilled", zorder=0, bins=np.arange(0, 10, step=1), align="left")

pmax = []
ax[0, 1].set_title("PEG")
for (i, (peg_data, snr)) in enumerate(zip([pegt_inf_smooth_pc, pegt_0_smooth_pc, pegt_5_smooth_pc], ["inf", "0dB", "-5dB"])):
    for j in range(peg_data.shape[0]):
        if np.isfinite(peg_data[j, 0]):
            ax[0, 1].plot(peg_data[j, :], color=peg_cmap(i+2), lw=0.5, zorder=-1)
    ax[0, 1].plot(np.nanmean(peg_data, axis=0), color=peg_cmap(i+2), lw=2, zorder=0, label=snr)
    pmax += list(np.argmax(peg_data, axis=1)[np.isfinite(peg_data[:, 0])])
ax[1, 1].hist(pmax, color="tab:orange", edgecolor="k", 
                histtype="stepfilled", zorder=0, bins=np.arange(0, 10, step=1), align="left")

for a in ax.flatten():
    a.axvline(3.5, color="k")
    a.axvspan(0.5, 4.5, color="grey", alpha=0.3, lw=0, zorder=-1)
    a.set_xticks(np.arange(0, len(twindows), step=2)+0.5)
    a.set_xticklabels(np.round(np.arange(0.1, 0.6, step=0.1), 2))

for a in [ax[0, 0], ax[0, 1]]:
    a.set_ylabel("percent correct")
    a.legend(frameon=False, bbox_to_anchor=(0, 1), loc="lower left")
    a.set_ylim((0.65, 1.0))
    a.set_xlabel("Time (s)")

for a in [ax[1, 0], ax[1, 1]]:
    a.set_ylabel("count")
    a.set_xlabel("Time of peak choice prob. (s)")


f.tight_layout()