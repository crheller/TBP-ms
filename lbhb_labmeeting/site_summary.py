"""
Summary per cell weights (u axis and noise axis)
Plot ellipse plot
Plot PSTHs sorted by weight
Plot example raster / PSTH plots for example neurons
"""
import charlieTools.TBP_ms.loaders as loaders
import charlieTools.TBP_ms.plotting as plotting
import charlieTools.TBP_ms.decoding as decoding
import nems0.db as nd
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb import tin_helpers as thelp

import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from settings import BAD_SITES
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': 12})

batch = 324
site = "CLT007a"
mask = ["HIT_TRIAL", "CORRECT_REJECT_TRIAL", "PASSIVE_EXPERIMENT", "MISS_TRIAL"]
figpath = "/auto/users/hellerc/code/projects/TBP-ms/temp_figs/ex_sites/"

# load data
amask = [m for m in mask if m!="PASSIVE_EXPERIMENT"]
pmask = ["PASSIVE_EXPERIMENT"]
Xa, _ = loaders.load_tbp_for_decoding(site=site, 
                                batch=batch,
                                wins = 0.1,
                                wine = 0.4,
                                collapse=True,
                                mask=amask,
                                recache=False)
Xp, _ = loaders.load_tbp_for_decoding(site=site, 
                                batch=batch,
                                wins = 0.1,
                                wine = 0.4,
                                collapse=True,
                                mask=pmask,
                                recache=False)

# get dDR space
# use both active / passive data. Compute u just using pure tone vs. catch
targets = [t for t in Xa.keys() if t.startswith("TAR")]
refs = [t for t in Xa.keys() if t.startswith("STIM")]
catch = [c for c in Xa.keys() if c.startswith("CAT")]
X = dict.fromkeys(targets+catch)

# get color maps
bwg, cm = thelp.make_tbp_colormaps(ref_stims=refs, tar_stims=targets+catch, use_tar_freq_idx=0)

for t in targets+catch:
    X[t] = np.concatenate((Xa[t], Xp[t]), axis=1)

cat = catch[0]
tar = [t for t in targets if "Inf" in t][0]
axes = decoding.get_decoding_space(X, [(tar, cat)], method="dDR", 
                        noise_space="targets")[0]

# plot weights
f, ax = plt.subplots(1, 1, figsize=(5, 3))

ax.plot(axes[0], "o-", label=r"$\Delta \mu$")
ax.plot(axes[1], "o-", label=r"$\sigma$")
ax.set_ylabel("Loading weight")
ax.set_xlabel("Neuron ID")
ax.legend(frameon=False, fontsize=10)

f.savefig(os.path.join(figpath, f"{site}_loadings.png"), dpi=200)

# plot ellipses
f, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

for ii, stim in enumerate(catch+targets):
    xa = Xa[stim][:, :, 0].T @ axes.T
    xp = Xp[stim][:, :, 0].T @ axes.T
    ax[0].scatter(xa[:, 0], xa[:, 1], c=cm(ii), alpha=0.5)
    el = plotting.cplt.compute_ellipse(xa[:, 0], xa[:, 1])
    ax[0].plot(el[0], el[1], lw=2, c=cm(ii))
    ax[1].scatter(xp[:, 0], xp[:, 1], c=cm(ii), alpha=0.5)
    el = plotting.cplt.compute_ellipse(xp[:, 0], xp[:, 1])
    ax[1].plot(el[0], el[1], lw=2, c=cm(ii))

ax[0].set_title("Active")
ax[1].set_title("Passive")
ax[0].set_xlabel(r"$\Delta \mu$"); ax[1].set_xlabel(r"$\Delta \mu$")
ax[0].set_ylabel(r"$\sigma$"); ax[1].set_ylabel(r"$\sigma$")

f.tight_layout()
f.savefig(os.path.join(figpath, f"{site}_ellipsePlot.png"), dpi=200)

# plot example PSTHs / rasters
rasterfs = 40
rasterfs2 = 1000
options = {'rasterfs': rasterfs, 'pupil': True, 'resp': True, 'stim': False}
options2 = {'rasterfs': rasterfs2, 'pupil': True, 'resp': True, 'stim': False}

# for psths
manager = BAPHYExperiment(cellid=site, batch=batch)
rec = manager.get_recording(**options)
rec['resp'] = rec['resp'].rasterize()
# for spike raster
manager2 = BAPHYExperiment(cellid=site, batch=batch)
rec2 = manager2.get_recording(**options2)
rec2['resp'] = rec2['resp'].rasterize()
st = 0.0
en = 0.5
bs = int(st * rasterfs)
be = int(en * rasterfs)
bsr = int(st * rasterfs2)
ber = int(en * rasterfs2)

# select cellids based on decoding space weights
cellids = np.array(rec["resp"].chans)[np.argsort(axes[0, :])[::-1]][:2]
ext = "mu_big"
ticksize = 3
for c in cellids:
    f, ax = plt.subplots(1, 1, figsize=(7, 4))

    for s in catch+targets:
        cidx = np.argwhere(s==np.array(catch+targets))[0][0]
        # PASSIVE
        p_resp = rec['resp'].extract_channels([c]).extract_epoch(s, mask=rec.and_mask(['PASSIVE_EXPERIMENT'])['mask'])[:, :, bs:be] * rasterfs
        t = np.linspace(0, 0.5, p_resp.shape[-1])
        psth_resp = p_resp.mean(axis=(0, 1))
        sem_resp = p_resp.std(axis=(0, 1)) / np.sqrt(p_resp.shape[0])
        ax.plot(t, psth_resp, lw=2, color=cm(cidx))
        ax.fill_between(t, psth_resp-sem_resp, psth_resp+sem_resp, color=cm(cidx), alpha=0.3, lw=0)

        # ACTIVE
        a_resp = rec['resp'].extract_channels([c]).extract_epoch(s, mask=rec.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])['mask'])[:, :, bs:be] * rasterfs
        psth_resp = a_resp.mean(axis=(0, 1))
        sem_resp = a_resp.std(axis=(0, 1)) / np.sqrt(a_resp.shape[0])
        ax.plot(t+t[-1]+0.1, psth_resp, lw=2, color=cm(cidx), label=s.strip('+NoiseTAR_CAT_'))
        ax.fill_between(t+t[-1]+0.1, psth_resp-sem_resp, psth_resp+sem_resp, color=cm(cidx), alpha=0.3, lw=0)

    offset = ax.get_ylim()[1] + 5
    rspan = int(offset / 3)
    for s in catch+targets:
        cidx = np.argwhere(s==np.array(catch+targets))[0][0]
        # PASSIVE
        rast = rec2['resp'].extract_channels([c]).extract_epoch(s, mask=rec2.and_mask(['PASSIVE_EXPERIMENT'])['mask'])[:, :, bsr:ber].squeeze()
        un, st = np.where(rast)
        st = st / rasterfs2
        un = un / (rast.shape[0] / rspan)
        un += offset
        ax.plot(st, un, '|', markersize=ticksize, alpha=0.4, color=cm(cidx))
        # ACTIVE
        rast = rec2['resp'].extract_channels([c]).extract_epoch(s, mask=rec2.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])['mask'])[:, :, bsr:ber].squeeze()
        un, st = np.where(rast)
        st = st / rasterfs2
        st += t[-1] + 0.1
        un = un / (rast.shape[0] / rspan)
        un += offset
        ax.plot(st, un, '|', markersize=ticksize, alpha=0.4, color=cm(cidx))

        offset += rspan + 3

    top = ax.get_ylim()[1]
    ax.text(0.2, top, 'PASSIVE', color='k')
    ax.text(0.8, top, 'ACTIVE', color='k')
    for vline in [0.1, 0.4, 0.7, 1.0]:
        ax.axvline(vline, linestyle='--', lw=0.5, color='k', zorder=-1)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False, fontsize=10)    
    ax.set_ylabel('Spike / sec', fontsize=10)
    ax.set_xlabel('Time (sec)', fontsize=10)     

    f.tight_layout()
    f.savefig(os.path.join(figpath, f"{c}_psthRaster_{ext}.png"), dpi=200)