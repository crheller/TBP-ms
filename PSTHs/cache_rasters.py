"""
For each cell, plot PSTH / raster plot for active vs. passive.
In active, include only CR / HIT trials
"""
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

figpath = "/auto/users/hellerc/code/projects/TBP-ms/temp_figs/"

batch = 324
rasterfs = 40
options = {'rasterfs': rasterfs, 'pupil': True, 'resp': True, 'stim': False}
wins = 0.0
wine = 0.5
bs = int(wins * rasterfs)
be = int(wine * rasterfs)

sites = nd.get_batch_sites(324)[0]
sites = [s for s in sites if s not in BAD_SITES]

for site in sites:
    manager = BAPHYExperiment(cellid=site, batch=batch)
    rec = manager.get_recording(**options)
    rec['resp'] = rec['resp'].rasterize()

    refs, tars, all_sounds = thelp.get_sound_labels(rec)
    bwg, tar = thelp.make_tbp_colormaps(ref_stims=refs, tar_stims=tars, use_tar_freq_idx=0)

    f, ax = plt.subplots(10, 5, figsize=(14, 14))
    title_string = ""
    for i, tt in enumerate(tars):
        try:
            ractive = rec['resp'].extract_epoch(tt, mask=rec.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])['mask'])[:, :, bs:be]
            rpassive = rec['resp'].extract_epoch(tt, mask=rec.and_mask(['PASSIVE_EXPERIMENT'])['mask'])[:, :, bs:be]

            title_string += f"{tt}: n={ractive.shape[0]} / {rpassive.shape[0]} ,"

            for c in range(ractive.shape[1]):
                cidx = np.argwhere(tt==np.array(tars))[0][0]
                # PASSIVE
                t = np.linspace(0, 0.5, rpassive.shape[-1])
                psth_resp = rpassive[:, c, :].mean(axis=0)
                sem_resp = rpassive[:, c, :].std(axis=0) / np.sqrt(rpassive.shape[0])
                ax.flatten()[c].plot(t, psth_resp, lw=2, color=tar(cidx))
                ax.flatten()[c].fill_between(t, psth_resp-sem_resp, psth_resp+sem_resp, color=tar(cidx), alpha=0.3, lw=0)

                # ACTIVE
                psth_resp = ractive[:, c, :].mean(axis=0)
                sem_resp = ractive[:, c, :].std(axis=0) / np.sqrt(ractive.shape[0])
                ax.flatten()[c].plot(t+t[-1]+0.1, psth_resp, lw=2, color=tar(cidx), label=tt.strip('+NoiseTAR_CAT_'))
                ax.flatten()[c].fill_between(t+t[-1]+0.1, psth_resp-sem_resp, psth_resp+sem_resp, color=tar(cidx), alpha=0.3, lw=0)

                ax.flatten()[c].axvline(0.55, linestyle="--", color="k", lw=3)
                ax.flatten()[c].set_xticks([])
                ax.flatten()[c].set_title(rec["resp"].chans[c])
        except IndexError:
            print(f"stim {tt} not present in active and passive")

    f.suptitle(title_string, y=1.0)
    f.tight_layout()
    f.savefig(os.path.join(figpath, f"{site}_psths.png"))