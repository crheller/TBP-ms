"""
Compute (pre-stim normalized) response for each neuron / target.
Mean of response in evoked window.

Cache results to load for figure 2.
"""
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems_lbhb.tin_helpers as thelp
import scipy.ndimage.filters as sf
import scipy.stats as ss
import sys
import nems0.db as nd
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from settings import BAD_SITES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8
mpl.rcParams['xtick.labelsize'] = 8 
mpl.rcParams['ytick.labelsize'] = 8 

figpath = "/auto/users/hellerc/code/projects/TBP-ms/figure_files/fig2/"
batch = 324
sites = [s for s in nd.get_batch_sites(batch)[0] if s not in BAD_SITES]
fs = 50
amask = ["HIT_TRIAL", "CORRECT_REJECT_TRIAL"]
pmask = ["PASSIVE_EXPERIMENT"]

dfs = []
for site in sites:
    area = nd.pd_query(sql="SELECT area from sCellFile where cellid like %s", params=(f"%{site}%",)).iloc[0][0]
    options = {'resp': True, 'pupil': True, 'rasterfs': fs, 'stim': False}
    manager = BAPHYExperiment(batch=batch, cellid=site, rawid=None)
    rec = manager.get_recording(recache=False, **options)
    rec['resp'] = rec['resp'].rasterize()
    rec = rec.create_mask(True)
    arec = rec.and_mask(amask)
    prec = rec.and_mask(pmask)

    ref, tars, _ = thelp.get_sound_labels(arec)

    for tar in tars:
        try:
            atresp = rec["resp"].extract_epoch(str(tar), mask=arec["mask"])
            ptresp = rec["resp"].extract_epoch(str(tar), mask=prec["mask"])

            # statistical test across trials
            pval = np.zeros(atresp.shape[1])
            for n in range(atresp.shape[1]):
                pval[n] = ss.ranksums(atresp[:, n, int(0.1*fs):int(0.4*fs)].mean(axis=-1), ptresp[:, n, int(0.1*fs):int(0.4*fs)].mean(axis=-1)).pvalue

            at_psth = atresp.mean(axis=0)[:, :int(0.5 * fs)]
            pt_psth = ptresp.mean(axis=0)[:, :int(0.5 * fs)]

            # normalize to prestim baseline
            at_psth = ((at_psth.T - at_psth[:, :int(0.1*fs)].mean(axis=1))).T
            pt_psth = ((pt_psth.T - pt_psth[:, :int(0.1*fs)].mean(axis=1))).T

            # get mean evoked response
            dur_s = 0.3
            aresp = np.sum(at_psth[:, int(0.1*fs):int(0.4*fs)], axis=1) * dur_s
            presp = np.sum(pt_psth[:, int(0.1*fs):int(0.4*fs)], axis=1) * dur_s

            df = pd.DataFrame(index=rec["resp"].chans,
                                data=np.vstack([aresp, presp, pval, [tar] * at_psth.shape[0], 
                                                [site] * at_psth.shape[0],
                                                [area] * at_psth.shape[0]]).T,
                                columns=["active", "passive", "pval", "epoch", "site", "area"])
            dfs.append(df)
        except:
            print(f"didn't find epoch: {tar}")

df = pd.concat(dfs)
df["snr"] = [v[1] for v in df["epoch"].str.split("+").values]
df = df.astype({
    "active": float,
    "passive": float,
    "pval": float,
    "snr": object,
    "area": object,
    "epoch": object,
    "site": object
})
df["cellid"] = df.index

df.to_csv("/auto/users/hellerc/results/TBP-ms/tar_vs_cat.csv")

gg = df.groupby(by=["snr", "cellid", "area"]).mean()

# PEG
f, ax = plt.subplots(4, 2, figsize=(4, 8), sharex=True, sharey=True)

ggm = gg[gg.index.get_level_values(2)=="PEG"]
snrs = ["-10dB", "-5dB", "0dB", "InfdB"]
for i, snr in enumerate(snrs):
    yvals = ggm[ggm.index.get_level_values(0)==snr]
    xvals = ggm[ggm.index.get_level_values(0)=="-InfdB"]
    vals = yvals.merge(xvals, on="cellid")

    ax[i, 0].scatter(
        vals["passive_y"],
        vals["passive_x"],
        s=25
    )
    ax[i, 1].scatter(
        vals["active_y"],
        vals["active_x"],
        s=25
    )
    ax[i, 0].set_ylabel(f"Target, {snr}")

mm = np.max(ax.flatten()[-1].get_ylim() + ax.flatten()[-1].get_xlim())
mi = np.min(ax.flatten()[-1].get_ylim() + ax.flatten()[-1].get_xlim())
for a in ax.flatten():
    a.plot([mi, mm], [mi, mm], "k--")
ax[0, 0].set_title("Passive")
ax[0, 1].set_title("Active")
ax[-1, 0].set_xlabel("Catch")
ax[-1, 1].set_xlabel("Catch")
f.tight_layout()

# A1
f, ax = plt.subplots(4, 2, figsize=(4, 8), sharex=True, sharey=True)

ggm = gg[gg.index.get_level_values(2)=="A1"]
snrs = ["-10dB", "-5dB", "0dB", "InfdB"]
for i, snr in enumerate(snrs):
    yvals = ggm[ggm.index.get_level_values(0)==snr]
    xvals = ggm[ggm.index.get_level_values(0)=="-InfdB"]
    vals = yvals.merge(xvals, on="cellid")

    ax[i, 0].scatter(
        vals["passive_y"],
        vals["passive_x"],
        s=25
    )
    ax[i, 1].scatter(
        vals["active_y"],
        vals["active_x"],
        s=25
    )
    ax[i, 0].set_ylabel(f"Target, {snr}")

mm = np.max(ax.flatten()[-1].get_ylim() + ax.flatten()[-1].get_xlim())
mi = np.min(ax.flatten()[-1].get_ylim() + ax.flatten()[-1].get_xlim())
for a in ax.flatten():
    a.plot([mi, mm], [mi, mm], "k--")
ax[0, 0].set_title("Passive")
ax[0, 1].set_title("Active")
ax[-1, 0].set_xlabel("Catch")
ax[-1, 1].set_xlabel("Catch")
f.tight_layout()