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
import nems_lbhb.tin_helpers as thelp
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
    options = {'resp': True, 'pupil': True, 'rasterfs': 30, 'stim': False}
    manager = BAPHYExperiment(batch=324, cellid=site, rawid=None)
    rec = manager.get_recording(recache=False, **options)
    rec['resp'] = rec['resp'].rasterize()
    ref, tar, all_stim = thelp.get_sound_labels(rec)

    max_pupil = rec["pupil"]._data.max()

    # Get pupil for specific behavioral outcomes / stimuli
    r = rec.copy()
    r = r.create_mask(True)
    r = r.and_mask(["HIT_TRIAL"])
    hit_trial = rec["pupil"].extract_epochs([t for t in tar if "TAR" in t], mask=r["mask"])
    hit_trial = np.concatenate([v[:, :, :15] for k, v in hit_trial.items()], axis=0)
    hit_trials.append(hit_trial / max_pupil)

    r = rec.copy()
    r = r.create_mask(True)
    r = r.and_mask(["MISS_TRIAL"])
    miss_trial = rec["pupil"].extract_epochs([t for t in tar if "TAR" in t], mask=r["mask"])
    miss_trial = np.concatenate([v[:, :, :15] for k, v in miss_trial.items()], axis=0)
    miss_trials.append(miss_trial / max_pupil)

    r = rec.copy()
    r = r.create_mask(True)
    r = r.and_mask(["CORRECT_REJECT_TRIAL"])
    cr_trial = rec["pupil"].extract_epochs([t for t in tar if "CAT" in t], mask=r["mask"])
    cr_trial = np.concatenate([v[:, :, :15] for k, v in cr_trial.items()], axis=0)
    cr_trials.append(cr_trial / max_pupil)

    r = rec.copy()
    r = r.create_mask(True)
    r = r.and_mask(["FALSE_ALARM_TRIAL"])
    fa_trial = rec["pupil"].extract_epochs(ref, mask=r["mask"])
    fa_trial = np.concatenate([v[:, :, :15] for k, v in fa_trial.items()], axis=0)
    fa_trials.append(fa_trial / max_pupil)

    r = rec.copy()
    r = r.create_mask(True)
    r = r.and_mask(["INCORRECT_HIT_TRIAL"])
    ich_trial = rec["pupil"].extract_epochs([t for t in tar if "CAT" in t], mask=r["mask"])
    ich_trial = np.concatenate([v[:, :, :15] for k, v in ich_trial.items()], axis=0)
    ich_trials.append(ich_trial / max_pupil)

hit_trials = np.concatenate(hit_trials, axis=0)
miss_trials = np.concatenate(miss_trials, axis=0)
fa_trials = np.concatenate(fa_trials, axis=0)
cr_trials = np.concatenate(cr_trials, axis=0)
ich_trials = np.concatenate(ich_trials, axis=0)


# Split active by behavioral output
f, ax = plt.subplots(1, 2, figsize=(10, 3))

t = np.linspace(0, 0.5, 15)
keys = ["hit", "miss", "false alarm", "correct reject", "incorrect hit"]
data = [hit_trials, miss_trials, fa_trials, cr_trials, ich_trials]
for (kk, mm) in zip(keys, data):
    u = np.nanmean(mm, axis=0)[0, :]
    sem = np.nanstd(mm, axis=0)[0, :] / np.sqrt(mm.shape[0])
    ax[0].plot(t, u, label=kk)
    ax[0].fill_between(t, u-sem, u+sem, alpha=0.3)

    mm = (mm.T - mm[:, 0, 0]).T
    u = np.nanmean(mm, axis=0)[0, :]
    sem = np.nanstd(mm, axis=0)[0, :] / np.sqrt(mm.shape[0])
    ax[1].plot(t, u, label=kk)
    ax[1].fill_between(t, u-sem, u+sem, alpha=0.3)

ax[0].set_ylabel(r"Pupil size (max$^{-1}$)")
ax[0].set_xlabel("Time (s)")

ax[1].set_ylabel(r"Pupil change (max$^{-1}$)")
ax[1].set_xlabel("Time (s)")

ax[1].legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")

f.tight_layout()