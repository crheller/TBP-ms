"""
Loop over datasets.
For each, load pupil.
Split pupil into trial types (HIT, CORRECT_REJECT, MISS, FALSE ALARM vs. PASSIVE)
Look at "target evoked" pupil on HIT / CORRECT_REJECT / MISS / PASSIVE
To compare with false alarm, need to look at per-trial pupil (e.g. as in Saderi)
    - or can look just at "INCORRECT_HITS" (that's a fair comparison with CORRECT_REJECT, I think)
"""
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

sites, cells = nd.get_batch_sites(324)

# chop off trials at 4 seconds
active_trials = []
passive_trials = []
fa_trials = []
hit_trials = []
miss_trials = []
cr_trials = []
ich_trials = []
for site in sites:
    options = {'resp': True, 'pupil': True, 'rasterfs': 10, 'stim': False}
    manager = BAPHYExperiment(batch=324, cellid=site, rawid=None)
    rec = manager.get_recording(recache=False, **options)
    rec['resp'] = rec['resp'].rasterize()

    max_pupil = rec["pupil"]._data.max()

    # get active pupil per trial
    ra = rec.copy()
    ra = ra.create_mask(True)
    ra = ra.and_mask(["CORRECT_REJECT_TRIAL", "INCORRECT_HIT_TRIAL", "FALSE_ALARM_TRIAL" "HIT_TRIAL", "MISS_TRIAL"])
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

    r = rec.copy()
    r = r.create_mask(True)
    r = r.and_mask(["INCORRECT_HIT_TRIAL"])
    ich_trial = rec["pupil"].extract_epochs("TRIAL", mask=r["mask"])
    ich_trials.append(ich_trial["TRIAL"][:, 0, :40] / max_pupil)

active_trials = np.concatenate(active_trials, axis=0)
passive_trials = np.concatenate(passive_trials, axis=0)
hit_trials = np.concatenate(hit_trials, axis=0)
miss_trials = np.concatenate(miss_trials, axis=0)
fa_trials = np.concatenate(fa_trials, axis=0)
cr_trials = np.concatenate(cr_trials, axis=0)
ich_trials = np.concatenate(ich_trials, axis=0)


# Split active vs. passive
f, ax = plt.subplots(1, 2, figsize=(10, 3))

t = np.linspace(0, 4, 40)
keys = ["active", "passive"]
data = [active_trials, passive_trials]
for (kk, mm) in zip(keys, data):
    u = np.nanmean(mm, axis=0)
    sem = np.nanstd(mm, axis=0) / np.sqrt(mm.shape[0])
    ax[0].plot(t, u, label=kk)
    ax[0].fill_between(t, u-sem, u+sem, alpha=0.3)

    mm = (mm.T - mm[:, 1]).T
    u = np.nanmean(mm, axis=0)
    sem = np.nanstd(mm, axis=0) / np.sqrt(mm.shape[0])
    ax[1].plot(t, u, label=kk)
    ax[1].fill_between(t, u-sem, u+sem, alpha=0.3)

ax[0].set_ylabel(r"Pupil size (max$^{-1}$)")
ax[0].set_xlabel("Time (s)")

ax[1].set_ylabel(r"Pupil change (max$^{-1}$)")
ax[1].set_xlabel("Time (s)")

ax[1].legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")

f.tight_layout()

# Split active by behavioral output
f, ax = plt.subplots(1, 2, figsize=(10, 3))

t = np.linspace(0, 4, 40)
keys = ["hit", "miss", "false alarm", "correct reject", "incorrect hit"]
data = [hit_trials, miss_trials, fa_trials, cr_trials, ich_trials]
for (kk, mm) in zip(keys, data):
    u = np.nanmean(mm, axis=0)
    sem = np.nanstd(mm, axis=0) / np.sqrt(mm.shape[0])
    ax[0].plot(t, u, label=kk)
    ax[0].fill_between(t, u-sem, u+sem, alpha=0.3)

    mm = (mm.T - mm[:, 0]).T
    u = np.nanmean(mm, axis=0)
    sem = np.nanstd(mm, axis=0) / np.sqrt(mm.shape[0])
    ax[1].plot(t, u, label=kk)
    ax[1].fill_between(t, u-sem, u+sem, alpha=0.3)

ax[0].set_ylabel(r"Pupil size (max$^{-1}$)")
ax[0].set_xlabel("Time (s)")

ax[1].set_ylabel(r"Pupil change (max$^{-1}$)")
ax[1].set_xlabel("Time (s)")

ax[1].legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")

f.tight_layout()