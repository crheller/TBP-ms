"""
Loop over datasets.
For each, load pupil.
Split pupil into trial types (HIT, CORRECT_REJECT, MISS, FALSE ALARM vs. PASSIVE)

Create supplemental figure that shows
    1. baseline pupil reflects behavior (so impulsivity, basically)
    1.2. baseline pupil shows inverted U (incorrect trials have either bigger or smaller pupil)
    2. delta pupil reflects reward (bigger delta for rewarded - correct, trials)

"""
from itertools import combinations
import os
import scipy.stats as ss
import numpy as np
import nems0.db as nd
import matplotlib.pyplot as plt
from nems_lbhb.baphy_experiment import BAPHYExperiment
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8
mpl.rcParams['xtick.labelsize'] = 8 
mpl.rcParams['ytick.labelsize'] = 8

figpath = "/auto/users/hellerc/code/projects/TBP-ms/figure_files/supplemental_pup_beh"

sites, cells = nd.get_batch_sites(324)

# chop off trials at 4 seconds
active_trials = []
passive_trials = []
fa_trials = []
hit_trials = []
miss_trials = []
cr_trials = []
for (i, site) in enumerate(sites):
    options = {'resp': True, 'pupil': True, 'rasterfs': 10, 'stim': False}
    manager = BAPHYExperiment(batch=324, cellid=site, rawid=None)
    rec = manager.get_recording(recache=False, **options)
    rec['resp'] = rec['resp'].rasterize()

    max_pupil = rec["pupil"]._data.max()

    # get active pupil per trial
    ra = rec.copy()
    ra = ra.create_mask(True)
    ra = ra.and_mask(["CORRECT_REJECT_TRIAL", "FALSE_ALARM_TRIAL" "HIT_TRIAL", "MISS_TRIAL"])
    pa_trial = ra["pupil"].extract_epochs("TRIAL", mask=ra["mask"])
    active_trials.append(pa_trial["TRIAL"][:, 0, :40] / max_pupil)

    # get passive pupil per trial
    rp = rec.copy()
    rp = rp.create_mask(True)
    rp = rp.and_mask(["PASSIVE_EXPERIMENT"])
    pp_trial = rec["pupil"].extract_epochs("TRIAL", mask=rp["mask"])
    passive_trials.append(pp_trial["TRIAL"][:, 0, :40] / max_pupil)

    # Do same for specific behavioral outcomes
    r = rec.copy()
    r = r.create_mask(True)
    r = r.and_mask(["HIT_TRIAL"])
    hit_trial = rec["pupil"].extract_epochs("TRIAL", mask=r["mask"])
    hit_trials.append(hit_trial["TRIAL"][:, 0, :40] / max_pupil)

    r = rec.copy()
    r = r.create_mask(True)
    r = r.and_mask(["MISS_TRIAL"])
    miss_trial = rec["pupil"].extract_epochs("TRIAL", mask=r["mask"])
    miss_trials.append(miss_trial["TRIAL"][:, 0, :40] / max_pupil)

    r = rec.copy()
    r = r.create_mask(True)
    r = r.and_mask(["CORRECT_REJECT_TRIAL"])
    cr_trial = rec["pupil"].extract_epochs("TRIAL", mask=r["mask"])
    cr_trials.append(cr_trial["TRIAL"][:, 0, :40] / max_pupil)

    r = rec.copy()
    r = r.create_mask(True)
    r = r.and_mask(["FALSE_ALARM_TRIAL"])
    fa_trial = rec["pupil"].extract_epochs("TRIAL", mask=r["mask"])
    fa_trials.append(fa_trial["TRIAL"][:, 0, :40] / max_pupil)

# concatenate means and trials across recording sites
active_mean = np.concatenate([a.mean(axis=0, keepdims=True) for a in active_trials], axis=0)
active_trials = np.concatenate(active_trials, axis=0)
passive_mean = np.concatenate([a.mean(axis=0, keepdims=True) for a in passive_trials], axis=0)
passive_trials = np.concatenate(passive_trials, axis=0)
hit_mean = np.concatenate([a.mean(axis=0, keepdims=True) for a in hit_trials], axis=0)
hit_trials = np.concatenate(hit_trials, axis=0)
miss_mean = np.concatenate([a.mean(axis=0, keepdims=True) for a in miss_trials], axis=0)
miss_trials = np.concatenate(miss_trials, axis=0)
fa_mean = np.concatenate([a.mean(axis=0, keepdims=True) for a in fa_trials], axis=0)
fa_trials = np.concatenate(fa_trials, axis=0)
cr_mean = np.concatenate([a.mean(axis=0, keepdims=True) for a in cr_trials], axis=0)
cr_trials = np.concatenate(cr_trials, axis=0)


# summary histogram of active vs. passive pupil (to make general arousal point)

# Split active by behavioral output
bw = (0, 5)
f, ax = plt.subplots(2, 3, figsize=(7, 4.5))

# active vs. passive pupil
counts, bins = np.histogram(passive_trials.flatten(), bins=np.arange(0, 1.05, step=0.05))
counts = counts / counts.sum()
ax[0, 0].hist(bins[:-1], bins, weights=counts, histtype="step", lw=2, label="passive")
counts, bins = np.histogram(active_trials.flatten(), bins=np.arange(0, 1, step=0.05))
counts = counts / counts.sum()
ax[0, 0].hist(bins[:-1], bins, weights=counts, histtype="step", lw=2, label="active")
ax[0, 0].set_ylabel("Fraction")
ax[0, 0].set_xlabel(r"Pupil size (max$^{-1}$)")
ax[0, 0].legend(frameon=False, bbox_to_anchor=(1, 1), loc="lower right")

yy0 = passive_mean[:, bw[0]:bw[1]].mean(axis=1)
yy1 = active_mean[:, bw[0]:bw[1]].mean(axis=1)
ax[1, 0].errorbar(0, yy0.mean(), yerr=yy0.std()/np.sqrt(len(yy0)),
                    marker="o", capsize=4, lw=2)
ax[1, 0].errorbar(1, yy1.mean(), yerr=yy1.std()/np.sqrt(len(yy0)),
                    marker="o", capsize=4, lw=2)
ax[1, 0].set_xlim((-1, 3))
ax[1, 0].set_ylabel(r"Pupil size (max$^{-1}$)")
ax[1, 0].set_xticks([])

# behavior pupil
t = np.linspace(0, 4, 40)
keys = ["hit", "correct reject", "miss", "false alarm"]
colors = ["darkblue", "cornflowerblue", "lightcoral", "firebrick"] # reds for incorrect, blues for correct
data_all = [hit_trials, cr_trials, miss_trials, fa_trials]
data = [hit_mean, cr_mean, miss_mean, fa_mean]
for i, (kk, mm, mmt, col) in enumerate(zip(keys, data, data_all, colors)):

    # plot in time
    u = np.nanmean(mmt, axis=0)
    sem = np.nanstd(mmt, axis=0) / np.sqrt(mmt.shape[0])
    ax[0, 1].plot(t, u, label=kk, color=col)
    ax[0, 1].fill_between(t, u-sem, u+sem, color=col, alpha=0.3, lw=0)

    # plot baseline summary
    yy = mm[:, bw[0]:bw[1]].mean(axis=1)
    ax[1, 1].errorbar(i, yy.mean(), yerr=yy.std()/np.sqrt(len(yy)),
                        marker="o", capsize=4, lw=2, c=col)

    mmt = (mmt.T - mmt[:, 0]).T
    u = np.nanmean(mmt, axis=0)
    sem = np.nanstd(mmt, axis=0) / np.sqrt(mmt.shape[0])
    ax[0, 2].plot(t, u, label=kk, color=col)
    ax[0, 2].fill_between(t, u-sem, u+sem, color=col, alpha=0.3, lw=0)

    mm = (mm.T - mm[:, 0]).T
    # plot delta summary
    yy = np.nanmax(mm, axis=1)
    ax[1, 2].errorbar(i, yy.mean(), yerr=yy.std()/np.sqrt(len(yy)),
                        marker="o", capsize=4, lw=2, c=col)

ax[0, 1].set_ylabel(r"Pupil size (max$^{-1}$)")
ax[0, 1].set_xlabel("Time (s)")
ax[1, 1].set_ylabel(r"Baseline pupil size (max$^{-1}$)")
ax[1, 1].set_xticks([])

ax[0, 2].set_ylabel(r"Pupil change (max$^{-1}$)")
ax[0, 2].set_xlabel("Time (s)")
ax[1, 2].set_ylabel(r"Max pupil change (max$^{-1}$)")
ax[1, 2].set_xticks([])

ax[0, 2].legend(frameon=False, bbox_to_anchor=(1, 1), loc="lower right")

f.tight_layout()
f.savefig(os.path.join(figpath, "behavior_pupil.svg"), dpi=500)

# pairwise stats
key_combos = list(combinations(keys, 2))
d_combos = list(combinations(data, 2))
for (kk, mm) in zip(key_combos, d_combos):
    # baseline
    pval, stat = ss.wilcoxon(mm[0][:, bw[0]:bw[1]].mean(axis=1), mm[1][:, bw[0]:bw[1]].mean(axis=1))
    print(f"basline {kk[0]} vs. {kk[1]}, pval: {pval}, stat: {stat}")
    # delta
    yy0 = (mm[0].T - mm[0][:, 0]).T
    yy0 = np.nanmax(yy0, axis=1)
    yy1 = (mm[1].T - mm[1][:, 0]).T
    yy1 = np.nanmax(yy1, axis=1)
    pval, stat = ss.wilcoxon(yy0, yy1)
    print(f"delta {kk[0]} vs. {kk[1]}, pval: {pval}, stat: {stat}\n")