"""
Draft figure with population PSTHs in active / passive and their difference
Create for one example site.
Show one Target response and one catch response
"""
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems_lbhb.tin_helpers as thelp
import scipy.ndimage.filters as sf
import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

site = "CRD010b"
batch = 324
fs = 50
amask = ["HIT_TRIAL", "CORRECT_REJECT_TRIAL"]
pmask = ["PASSIVE_EXPERIMENT"]

options = {'resp': True, 'pupil': True, 'rasterfs': fs, 'stim': False}
manager = BAPHYExperiment(batch=batch, cellid=site, rawid=None)
rec = manager.get_recording(recache=False, **options)
rec['resp'] = rec['resp'].rasterize()
rec = rec.create_mask(True)
arec = rec.and_mask(amask)
prec = rec.and_mask(pmask)

ref, tars, _ = thelp.get_sound_labels(rec)
target = tars[-1]
catch = tars[0]

atresp = rec["resp"].extract_epoch(target, mask=arec["mask"])
ptresp = rec["resp"].extract_epoch(target, mask=prec["mask"])
acresp = rec["resp"].extract_epoch(catch, mask=arec["mask"])
pcresp = rec["resp"].extract_epoch(catch, mask=prec["mask"])

sigma = 0.001
at_psth = sf.gaussian_filter1d(atresp.mean(axis=0)[:, :int(0.5 * fs)], sigma, axis=1)
pt_psth = sf.gaussian_filter1d(ptresp.mean(axis=0)[:, :int(0.5 * fs)], sigma, axis=1)
ac_psth = sf.gaussian_filter1d(acresp.mean(axis=0)[:, :int(0.5 * fs)], sigma, axis=1)
pc_psth = sf.gaussian_filter1d(pcresp.mean(axis=0)[:, :int(0.5 * fs)], sigma, axis=1)

# normalize according to all the data
m = np.concatenate((at_psth, pt_psth, ac_psth, pc_psth), axis=1).min(axis=1)
sd = np.concatenate((at_psth, pt_psth, ac_psth, pc_psth), axis=1).std(axis=1)
sd[sd==0] = 1
at_psth = ((at_psth.T - m) / sd).T
pt_psth = ((pt_psth.T - m) / sd).T
ac_psth = ((ac_psth.T - m) / sd).T
pc_psth = ((pc_psth.T - m) / sd).T

vmin = 0
vmax = 5
sidx = np.argsort(at_psth.argmax(axis=1))
f, ax = plt.subplots(2, 3, figsize=(7, 6))

ax[0, 0].set_title("Active, Target")
ii=ax[0, 0].imshow(at_psth[sidx, :], cmap="hot", vmin=vmin, vmax=vmax,
        aspect="auto", extent=[0, 0.5, 0, at_psth.shape[0]])
f.colorbar(ii, ax=ax[0, 0])
ax[0, 1].set_title("Passive, Target")
ii=ax[0, 1].imshow(pt_psth[sidx, :], cmap="hot", vmin=vmin, vmax=vmax,
        aspect="auto", extent=[0, 0.5, 0, at_psth.shape[0]])
f.colorbar(ii, ax=ax[0, 1])
ii=ax[0, 2].imshow(at_psth[sidx, :] - pt_psth[sidx, :], cmap="bwr", vmin=-vmax, vmax=vmax, 
        aspect="auto", extent=[0, 0.5, 0, at_psth.shape[0]])
f.colorbar(ii, ax=ax[0, 2])

ax[1, 0].set_title("Active, Catch")
ii=ax[1, 0].imshow(ac_psth[sidx, :], cmap="hot", vmin=vmin, vmax=vmax,
        aspect="auto", extent=[0, 0.5, 0, at_psth.shape[0]])
f.colorbar(ii, ax=ax[1, 0])
ax[1, 1].set_title("Passive, Catch")
ii=ax[1, 1].imshow(pc_psth[sidx, :], cmap="hot", vmin=vmin, vmax=vmax,
        aspect="auto", extent=[0, 0.5, 0, at_psth.shape[0]])
f.colorbar(ii, ax=ax[1, 1])
ii=ax[1, 2].imshow(ac_psth[sidx, :] - pc_psth[sidx, :], cmap="bwr", vmin=-vmax, vmax=vmax, 
        aspect="auto", extent=[0, 0.5, 0, at_psth.shape[0]])
f.colorbar(ii, ax=ax[1, 2])

for i in range(2):
    for j in range(3):
        if j != 2:
            c = "white"
        else:
            c = "k"

        ax[i, j].axvline(0.1, linestyle="--", color=c)
        ax[i, j].axvline(0.4, linestyle="--", color=c)
        ax[i, j].set_xlabel("Time (s)")

f.tight_layout()