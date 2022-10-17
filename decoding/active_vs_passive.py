"""
Simple script to load active vs. passive decding,
sort them based on category, and plot per site
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

batch = 324
sqrt = True
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])
amodel = 'tbpDecoding_mask.h.cr_drmask.h.cr.pa_DRops.dim2.ddr-targetNoise'
pmodel = 'tbpDecoding_mask.pa_drmask.h.cr.pa_DRops.dim2.ddr-targetNoise'

# as a "control" / "cross-validation", use results from dec. axis trained on opp. state
# amodel = 'tbpDecoding_mask.h.cr.m_decmask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'
# pmodel = 'tbpDecoding_mask.pa_decmask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'

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

df["f1"] = thelp.get_freqs(df["e1"].tolist())
df["f2"] = thelp.get_freqs(df["e2"].tolist())
df["oct_diff"] = abs(np.log2(df["f1"] / df["f2"]))
df["snr1"] = [thelp.get_snrs([e])[0] if e.startswith("STIM_")==False else -np.inf for e in df["e1"]]
df["snr1"][df.snr1==np.inf] = 10
df["snr1"][df.snr1==-np.inf] = -20
df["snr2"] = [thelp.get_snrs([e])[0] if e.startswith("STIM_")==False else -np.inf for e in df["e2"]]
df["snr2"][df.snr2==np.inf] = 10
df["snr2"][df.snr2==-np.inf] = -20

# plot summary per site
f, ax = plt.subplots(2, 4, figsize=(20, 10))

# A1
cm = (df["class"] == "tar_tar") & (df["area"]=="A1")
g = df[["dp_x", "dp_y",  "delta", "delta_raw", "class", "area", "site"]][cm].groupby(by="site").median()
ax[0, 0].scatter(g["dp_x"], g["dp_y"], color="red", edgecolor="k", s=50)
mm = max(ax[0, 0].get_xlim()+ax[0, 0].get_ylim())
ax[0, 0].plot([0, mm], [0, mm], "k--")
ax[0, 0].set_title("TAR vs. TAR, A1")
tt_diff = g["delta_raw"]
tt_diff_norm = g["delta"]

cm = (df["class"] == "ref_ref") & (df["area"]=="A1")
g = df[["dp_x", "dp_y",  "delta", "delta_raw", "class", "area", "site"]][cm].groupby(by="site").median()
ax[0, 1].scatter(g["dp_x"], g["dp_y"], color="blue", edgecolor="k", s=50)
mm = max(ax[0, 1].get_xlim()+ax[0, 1].get_ylim())
ax[0, 1].plot([0, mm], [0, mm], "k--")
ax[0, 1].set_title("REF vs. REF, A1")
rr_diff = g["delta_raw"]
rr_diff_norm = g["delta"]

cm = (df["class"] == "tar_cat") & (df["area"]=="A1")
g = df[["dp_x", "dp_y",  "delta", "delta_raw", "class", "area", "site"]][cm].groupby(by="site").median()
ax[0, 2].scatter(g["dp_x"], g["dp_y"], color="grey", edgecolor="k", s=50)
mm = max(ax[0, 2].get_xlim()+ax[0, 2].get_ylim())
ax[0, 2].plot([0, mm], [0, mm], "k--")
ax[0, 2].set_title("TAR vs. CAT, A1")
tc_diff = g["delta_raw"]
tc_diff_norm = g["delta"]

ri = np.random.normal(0, 0.1, len(tt_diff))
ax[0, 3].scatter([0], tt_diff_norm.median(), color="red", edgecolor="k", s=75)
ax[0, 3].scatter(ri, tt_diff_norm, alpha=0.5, s=25, color="red")
ri = np.random.normal(1, 0.1, len(rr_diff))
ax[0, 3].scatter([1], rr_diff_norm.median(), color="blue", edgecolor="k", s=75)
ax[0, 3].scatter(ri, rr_diff_norm, alpha=0.5, s=25, color="blue")
ri = np.random.normal(2, 0.1, len(tc_diff))
ax[0, 3].scatter([2], tc_diff_norm.median(), color="grey", edgecolor="k", s=75)
ax[0, 3].scatter(ri, tc_diff_norm, alpha=0.5, s=25, color="grey")
ax[0, 3].axhline(0, linestyle="--", color="k")

loc = ax[0, 3].get_ylim()[-1]
for i, dd in zip([0, 1, 2], [tt_diff_norm, rr_diff_norm, tc_diff_norm]):
    ax[0, 3].text(i-0.2, loc, f"p={round(ss.wilcoxon(dd).pvalue, 3)}")

# PEG
cm = (df["class"] == "tar_tar") & (df["area"]=="PEG")
g = df[["dp_x", "dp_y",  "delta", "delta_raw", "class", "area", "site"]][cm].groupby(by="site").median()
ax[1, 0].scatter(g["dp_x"], g["dp_y"], color="red", edgecolor="k", s=50)
mm = max(ax[1, 0].get_xlim()+ax[1, 0].get_ylim())
ax[1, 0].plot([0, mm], [0, mm], "k--")
ax[1, 0].set_title("TAR vs. TAR, PEG")
tt_diff = g["delta_raw"]
tt_diff_norm = g["delta"]

cm = (df["class"] == "ref_ref") & (df["area"]=="PEG")
g = df[["dp_x", "dp_y",  "delta", "delta_raw", "class", "area", "site"]][cm].groupby(by="site").median()
ax[1, 1].scatter(g["dp_x"], g["dp_y"], color="blue", edgecolor="k", s=50)
mm = max(ax[1, 1].get_xlim()+ax[1, 1].get_ylim())
ax[1, 1].plot([0, mm], [0, mm], "k--")
ax[1, 1].set_title("REF vs. REF, PEG")
rr_diff = g["delta_raw"]
rr_diff_norm = g["delta"]

cm = (df["class"] == "tar_cat") & (df["area"]=="PEG")
g = df[["dp_x", "dp_y",  "delta", "delta_raw", "class", "area", "site"]][cm].groupby(by="site").median()
ax[1, 2].scatter(g["dp_x"], g["dp_y"], color="grey", edgecolor="k", s=50)
mm = max(ax[1, 2].get_xlim()+ax[1, 2].get_ylim())
ax[1, 2].plot([0, mm], [0, mm], "k--")
ax[1, 2].set_title("TAR vs. CAT, PEG")
tc_diff = g["delta_raw"]
tc_diff_norm = g["delta"]


ri = np.random.normal(0, 0.1, len(tt_diff))
ax[1, 3].scatter([0], tt_diff_norm.median(), color="red", edgecolor="k", s=75)
ax[1, 3].scatter(ri, tt_diff_norm, alpha=0.5, s=25, color="red")
ri = np.random.normal(1, 0.1, len(rr_diff))
ax[1, 3].scatter([1], rr_diff_norm.median(), color="blue", edgecolor="k", s=75)
ax[1, 3].scatter(ri, rr_diff_norm, alpha=0.5, s=25, color="blue")
ri = np.random.normal(2, 0.1, len(tc_diff))
ax[1, 3].scatter([2], tc_diff_norm.median(), color="grey", edgecolor="k", s=75)
ax[1, 3].scatter(ri, tc_diff_norm, alpha=0.5, s=25, color="grey")
ax[1, 3].axhline(0, linestyle="--", color="k")

loc = ax[1, 3].get_ylim()[-1]
for i, dd in zip([0, 1, 2], [tt_diff_norm, rr_diff_norm, tc_diff_norm]):
    ax[1, 3].text(i-0.2, loc, f"p={round(ss.wilcoxon(dd).pvalue, 3)}")

f.tight_layout()

# delta dprime per site, compare tar vs. cat to tar vs. tar directly in PEG / A1
# TAR vs. TAR
cm = (df["class"] == "tar_tar") & (df["area"]=="A1")
tt_a1 = df[["dp_x", "dp_y", "delta", "delta_raw", "class", "area", "site"]][cm].groupby(by="site").median()
cm = (df["class"] == "tar_tar") & (df["area"]=="PEG")
tt_peg = df[["dp_x", "dp_y", "delta", "delta_raw", "class", "area", "site"]][cm].groupby(by="site").median()

# CAT vs. TAR
cm = (df["class"] == "tar_cat") & (df["area"]=="A1")
tc_a1 = df[["dp_x", "dp_y", "delta", "delta_raw", "class", "area", "site"]][cm].groupby(by="site").median()
cm = (df["class"] == "tar_cat") & (df["area"]=="PEG")
tc_peg = df[["dp_x", "dp_y", "delta", "delta_raw", "class", "area", "site"]][cm].groupby(by="site").median()

tt_a1 = tt_a1.loc[tc_a1.index]
tt_peg = tt_peg.loc[tc_peg.index]

f, ax = plt.subplots(2, 2, figsize=(10, 10))

# A1
ax[0, 0].scatter(tt_a1["delta"], tc_a1["delta"], color="k", s=75, edgecolor="white")
ax[0, 0].plot([-0.2, 0.5], [-0.2, 0.5], "--", color="grey")
ax[0, 0].axvline(0, linestyle="--", color="grey")
ax[0, 0].axhline(0, linestyle="--", color="grey")
ax[0, 0].set_title(f"A1 (norm.), p={round(ss.wilcoxon(tt_a1['delta'], tc_a1['delta']).pvalue, 3)}")

# PEG
ax[0, 1].scatter(tt_peg["delta"], tc_peg["delta"], color="k", s=75, edgecolor="white")
ax[0, 1].plot([-0.2, 0.5], [-0.2, 0.5], "--", color="grey")
ax[0, 1].axvline(0, linestyle="--", color="grey")
ax[0, 1].axhline(0, linestyle="--", color="grey")
ax[0, 1].set_title(f"PEG (norm.), p={round(ss.wilcoxon(tt_peg['delta'], tc_peg['delta']).pvalue, 3)}")

ax[1, 0].scatter(tt_a1["delta_raw"], tc_a1["delta_raw"], color="k", s=75, edgecolor="white")
ax[1, 0].plot([-2, 3], [-2, 3], "--", color="grey")
ax[1, 0].axvline(0, linestyle="--", color="grey")
ax[1, 0].axhline(0, linestyle="--", color="grey")
ax[1, 0].set_title(f"A1 (raw), p={round(ss.wilcoxon(tt_a1['delta_raw'], tc_a1['delta_raw']).pvalue, 3)}")


# PEG
ax[1, 1].scatter(tt_peg["delta_raw"], tc_peg["delta_raw"], color="k", s=75, edgecolor="white")
ax[1, 1].plot([-2, 3], [-2, 3], "--", color="grey")
ax[1, 1].axvline(0, linestyle="--", color="grey")
ax[1, 1].axhline(0, linestyle="--", color="grey")
ax[1, 1].set_title(f"PEG (raw), p={round(ss.wilcoxon(tt_peg['delta_raw'], tc_peg['delta_raw']).pvalue, 3)}")


for a in ax.flatten():
    a.set_ylabel(r"TAR vs. CAT $\Delta d'$")
    a.set_xlabel(r"TAR vs. TAR $\Delta d'$")

f.tight_layout()

# same as above, but with a paired line plot
f, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

for s in tt_a1.index:
    ax[0].plot([0, 1], [tt_a1.loc[s, "delta"], tc_a1.loc[s, "delta"]], "grey", lw=0.5)
ax[0].errorbar([0, 1], [tt_a1["delta"].median(), tc_a1["delta"].median()],
            yerr = [tt_a1["delta"].sem(), tc_a1["delta"].sem()],
            marker="o", color="k")
ax[0].axhline(0, linestyle="--", color="k", zorder=-1)
ax[0].set_xlim((-0.5, 1.5))
ax[0].set_ylabel(r"$\Delta d'$ (active vs. passive)")
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(["TAR vs. TAR", "TAR vs. CAT"])
ax[0].set_title("A1")

for s in tt_peg.index:
    ax[1].plot([0, 1], [tt_peg.loc[s, "delta"], tc_peg.loc[s, "delta"]], "grey", lw=0.5)
ax[1].errorbar([0, 1], [tt_peg["delta"].median(), tc_peg["delta"].median()],
            yerr = [tt_peg["delta"].sem(), tc_peg["delta"].sem()],
            marker="o", color="k")
ax[1].axhline(0, linestyle="--", color="k", zorder=-1)
ax[1].set_xlim((-0.5, 1.5))
ax[1].set_ylabel(r"$\Delta d'$ (active vs. passive)")
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(["TAR vs. TAR", "TAR vs. CAT"])
ax[1].set_title("PEG")

loc = ax[1].get_ylim()[-1]-0.05
ax[0].text(0.3, loc, f"p={round(ss.wilcoxon(tt_a1['delta'], tc_a1['delta']).pvalue, 3)}")
ax[1].text(0.3, loc, f"p={round(ss.wilcoxon(tt_peg['delta'], tc_peg['delta']).pvalue, 3)}")

f.tight_layout()


# ============== More fine grained analysis ===================
# break up results on per stim basis, sorted into SNR diff and freq diff (for ref / ref)


# REF vs. REF
# split up active vs. passive differences as fn of freq. diff and freq diff from target

# TAR vs. TAR
# active vs. passive difference as fn of SNR diff / abs SNR (of highest in the pair)
f, ax = plt.subplots(1, 2, figsize=(10, 5))

cm = (df["class"] == "tar_tar") & (df["area"]=="A1")
s1 = df[cm]["snr1"] + np.random.normal(0, 1, sum(cm))
s2 = df[cm]["snr2"] + np.random.normal(0, 1, sum(cm))
val = df[cm]["delta"]
ax[0].scatter(s1, s2, c=val, cmap="bwr", vmin=-0.5, vmax=0.5, s=50, edgecolor="k")
ax[0].set_xlabel("SNR 1")
ax[0].set_ylabel("SNR 2")
ax[0].plot([-10, 10], [-10, 10], "k--")
ax[0].set_title("A1")

cm = (df["class"] == "tar_tar") & (df["area"]=="PEG")
s1 = df[cm]["snr1"] + np.random.normal(0, 1, sum(cm))
s2 = df[cm]["snr2"] + np.random.normal(0, 1, sum(cm))
val = df[cm]["delta"]
ax[1].scatter(s1, s2, c=val, cmap="bwr", vmin=-0.5, vmax=0.5, s=50, edgecolor="k")
ax[1].set_xlabel("SNR 1")
ax[1].set_ylabel("SNR 2")
ax[1].plot([-10, 10], [-10, 10], "k--")
ax[1].set_title("PEG")

f.tight_layout()

