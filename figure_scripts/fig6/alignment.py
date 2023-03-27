"""
Compare first FA loading to the delta mu axis
for each target (vs. catch)
"""
import nems0.db as nd

import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
import os
import sys
from path_helpers import results_file
from settings import RESULTS_DIR, BAD_SITES

import scipy.stats as ss
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8 

figpath = "/auto/users/hellerc/code/projects/TBP-ms/figure_files/fig6/"

batch = 324
sqrt = True
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])
amodel = 'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'
pmodel = 'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'

sites = [s for s in sites if s not in BAD_SITES]
rra = 0
rrp = 0
dfa = pd.DataFrame(columns=["acos_sim", "e2", "e1", "area", "site"])
dfp = pd.DataFrame(columns=["pcos_sim", "e2", "e1", "area", "site"])
for site in sites:
    d = pd.read_pickle(os.path.join(RESULTS_DIR, "factor_analysis", str(batch), site, "FA_perstim.pickle"))
    area = nd.pd_query(sql="SELECT area from sCellFile where cellid like %s", params=(f"%{site}%",)).iloc[0][0]
    
    # load decoding results
    ares = pd.read_pickle(results_file(RESULTS_DIR, site, batch, amodel, "output.pickle"))
    pres = pd.read_pickle(results_file(RESULTS_DIR, site, batch, pmodel, "output.pickle"))
    for e in [k for k in d["active"].keys() if 'CAT' in k]:
        afa = d["active"][e]["components_"][0, :]
        pfa = d["passive"][e]["components_"][0, :]
        try:
            tars = ares[(ares["e1"]==e) & (ares["e2"].str.startswith("TAR"))].e2
            for tar in tars:
                mm = (ares["e1"]==e) & (ares["e2"]==tar)
                _du = ares[mm]["dU"].iloc[0]
                dua = (_du / np.linalg.norm(_du)).dot(ares[mm]["dr_loadings"].iloc[0]).squeeze()
                dfa.loc[rra, :]  = [np.abs(afa.dot(dua)), e, ares[mm]["e2"].iloc[0], area, site]
                rra += 1

            tars = ares[(ares["e1"]==e) & (ares["e2"].str.startswith("TAR"))].e2
            for tar in tars:
                mm = (ares["e1"]==e) & (ares["e2"]==tar)
                _dup = ares[mm]["dU"].iloc[0]
                dup = (_dup / np.linalg.norm(_dup)).dot(ares[mm]["dr_loadings"].iloc[0]).squeeze()
                dfp.loc[rrp, :]  = [np.abs(pfa.dot(dup)), e, ares[mm]["e2"].iloc[0], area, site]
                rrp += 1

        except IndexError:
            print(f"didn't find matching decoding entry for {e}, {site}")
    
# merge 
df = dfa.merge(dfp, on=["e1", "e2", "area", "site"])


f, ax = plt.subplots(1, 2, figsize=(2, 2), sharey=True)

for i, a in enumerate(["A1", "PEG"]):
    y = y = df[df.area==a]["pcos_sim"]
    ax[i].errorbar(0, y.mean(), yerr=y.std()/np.sqrt(len(y)), marker="o",
            capsize=2, markeredgecolor="k", label="passive") 
    y = df[df.area==a]["acos_sim"]
    ax[i].errorbar(1, y.mean(), yerr=y.std()/np.sqrt(len(y)), marker="o",
            capsize=2, markeredgecolor="k", label="active") 
    # ax[i].set_title(a)
    ax[i].set_xticks([])
    ax[i].set_xlim((-0.25, 1.25))
# ax[0].set_ylabel("Cos. similarity (dU vs. FA1)")
# ax[0].legend(frameon=False)
f.tight_layout()

f.savefig(os.path.join(figpath, "alignment_errorbar.svg"), dpi=500)

# pvalues
a1pval = ss.wilcoxon(df[df.area=="A1"]["acos_sim"], df[df.area=="A1"]["pcos_sim"])
pegpval = ss.wilcoxon(df[df.area=="PEG"]["acos_sim"], df[df.area=="PEG"]["pcos_sim"])
print(f"a1 alignement pval: {a1pval.pvalue}")
print(f"peg alignement pval: {pegpval.pvalue}")


# =====================================================================
# relationship between behavior and noise vs. decoding alignment
beh_df = pd.read_pickle(os.path.join(RESULTS_DIR, "behavior_recordings", "all_trials.pickle"))
# Plot relationship between behavior and neural dprime
bg = beh_df.groupby(by=["site", "e1"]).mean()
bg = bg.reset_index()
bg["e1"] = ["TAR_"+e for e in bg["e1"]]
merge = df.merge(bg, on=["e1","site"])
merge["delta"] = merge["acos_sim"] #- merge["pcos_sim"]
merge["snr"] = [float(snr[1].strip("dB")) for snr in merge["e1"].str.split("+")]
merge["snr"] = [snr if snr!=np.inf else 10 for snr in merge["snr"]]
merge = merge.astype({
    "delta": float,
    "dprime": float
})
a1_merge = merge[merge.area=="A1"]
peg_merge = merge[merge.area=="PEG"]

# ANOVA -- does SNR or behavior explain the change in alignment
import statsmodels.api as sm
Y = a1_merge["dprime"]
X = a1_merge[["snr", "delta"]]
X = (X - X.mean(axis=0)) / X.std(axis=0)
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
results.summary()

nboots = 500
s = 20
delta_ylim = (None, None) #(-0.25, 1)
colors = ["snr", "snr"] #["grey", "k"]
delta_metric = "delta"

f, ax = plt.subplots(1, 2, figsize=(4, 2))

for i, (_df, c) in enumerate(zip([a1_merge, peg_merge], colors)):

    x = _df["dprime"]
    xp = np.linspace(np.min(x), np.max(x), 100)
    r, p = ss.pearsonr(x, _df[delta_metric])
    leg = f"r={round(r, 3)}, p={round(p, 3)}"
    if c=="snr":
        sidx = np.argsort(_df["snr"]).values
        ax[i].scatter(x.values[sidx], _df[delta_metric].values[sidx], 
                        s=s, c=_df["snr"].values[sidx], 
                        vmin=-15, vmax=5, cmap="Reds", edgecolor="none", lw=0)
    else:
        ax[i].scatter(x, _df[delta_metric], 
                s=s, c=c, edgecolor="none", lw=0)

    # get line of best fit
    z = np.polyfit(x, _df[delta_metric], 1)
    # plot line of best fit
    p_y = z[1] + z[0] * xp
    if c=="snr":
        ax[i].plot(xp, p_y, lw=2, color="k")
    else:
        ax[i].plot(xp, p_y, lw=2, color=c)
    
    # bootstrap condifence interval
    boot_preds = []
    for bb in range(nboots):
        ii = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
        zb = np.polyfit(x.values[ii], _df[delta_metric].values[ii], 1)
        p_yb = zb[1] + zb[0] * xp
        boot_preds.append(p_yb)
    bse = np.stack(boot_preds).std(axis=0)
    lower = p_y - bse
    upper = p_y + bse
    if c=="snr":
        ax[i].fill_between(xp, lower, upper, color="k", alpha=0.5, lw=0)
    else:
        ax[i].fill_between(xp, lower, upper, color=c, alpha=0.5, lw=0)

    ax[i].set_ylim(delta_ylim)

f.tight_layout()

f.savefig(os.path.join(figpath, "delta_vs_behavior.svg"), dpi=500)

# compute bootstrapped p-values
np.random.seed(123)
nboots = 1000
x = a1_merge["dprime"].values
rb_a1_null = []
for bb in range(nboots):
    ii = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    jj = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    rb_a1_null.append(np.corrcoef(x[ii], a1_merge[delta_metric].values[jj])[0, 1])
x = peg_merge["dprime"].values
rb_peg_null = []
for bb in range(nboots):
    ii = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    jj = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    rb_peg_null.append(np.corrcoef(x[ii], peg_merge[delta_metric].values[jj])[0, 1])
a1_pval = np.mean(np.array(rb_a1_null) > np.corrcoef(a1_merge["dprime"], a1_merge[delta_metric].values)[0, 1])
peg_pval = np.mean(np.array(rb_peg_null) > np.corrcoef(peg_merge["dprime"], peg_merge[delta_metric].values)[0, 1])
print(f"A1 pval: {a1_pval}")
print(f"PEG pval: {peg_pval}")

# generate bootstrapped distro of cc values
np.random.seed(123)
nboots = 1000
x = a1_merge["dprime"]
rb_a1 = []
for bb in range(nboots):
    ii = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    rb_a1.append(np.corrcoef(x.values[ii], a1_merge[delta_metric].values[ii])[0, 1])
x = peg_merge["dprime"]
rb_peg = []
for bb in range(nboots):
    ii = np.random.choice(np.arange(0, len(x)), len(x), replace=True)
    rb_peg.append(np.corrcoef(x.values[ii], peg_merge[delta_metric].values[ii])[0, 1])

f, ax = plt.subplots(1, 1, figsize=(1, 2))

lower = np.quantile(rb_a1, 0.025)
upper = np.quantile(rb_a1, 0.975)
ax.plot([0, 0], [lower, upper], color="grey", zorder=-1)
ax.scatter([0], [np.mean(rb_a1)], s=50, edgecolor="k", c="grey")

lower = np.quantile(rb_peg, 0.025)
upper = np.quantile(rb_peg, 0.975)
ax.plot([1, 1], [lower, upper], color="k")
ax.scatter([1], [np.mean(rb_peg)], s=50, edgecolor="k", c="k")

ax.axhline(0, linestyle="--", color="grey")

ax.set_xlim((-0.1, 1.1))
ax.set_xticks([])

f.savefig(os.path.join(figpath, "pearson_95conf.svg"), dpi=500)
