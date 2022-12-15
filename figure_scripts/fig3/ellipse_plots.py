"""
Plot example ellipse plot(s) in dDR space for A1 and dPEG
    each 2x2 inches
"""

import os
import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from settings import RESULTS_DIR, BAD_SITES
from path_helpers import results_file

import charlieTools.TBP_ms.loaders as loaders
import charlieTools.plotting as cplt
import nems0.db as nd
import nems_lbhb.tin_helpers as thelp
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8
mpl.rcParams['xtick.labelsize'] = 8 
mpl.rcParams['ytick.labelsize'] = 8 

figpath = "/auto/users/hellerc/code/projects/TBP-ms/figure_files/fig3/"

batch = 324
sqrt = True  # d', not d'^2
fmodel = 'tbpDecoding_mask.pa_decmask.h.cr.m.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise-sharedSpace'

# load data from two example sites (one A1, one PEG)
a1_site = "CRD016c"
a1p, _ = loaders.load_tbp_for_decoding(site=a1_site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=["PASSIVE_EXPERIMENT"],
                                    recache=False)
a1a, _ = loaders.load_tbp_for_decoding(site=a1_site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=["HIT_TRIAL", "CORRECT_REJECT_TRIAL", "MISS_TRIAL"],
                                    recache=False)
peg_site = "CRD010b"
pegp, _ = loaders.load_tbp_for_decoding(site=peg_site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=["PASSIVE_EXPERIMENT"],
                                    recache=False)
pega, _ = loaders.load_tbp_for_decoding(site=peg_site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=["HIT_TRIAL", "CORRECT_REJECT_TRIAL", "MISS_TRIAL"],
                                    recache=False)

# get (fixed) dDR space for each site
dd = pd.read_pickle(results_file(RESULTS_DIR, a1_site, batch, fmodel, "output.pickle"))
a1_loading = dd["dr_loadings"].iloc[0]
dd = pd.read_pickle(results_file(RESULTS_DIR, peg_site, batch, fmodel, "output.pickle"))
peg_loading = dd["dr_loadings"].iloc[0]

## Plot ellipse plots
s = 10
f, ax = plt.subplots(2, 2, figsize=(4, 4))

# A1
catch = [c for c in a1a.keys() if "CAT_" in c]
targets = catch+[t for t in a1a.keys() if "TAR_" in t]
BwG, gR = thelp.make_tbp_colormaps(catch, targets, use_tar_freq_idx=0)
for i, t in enumerate(targets):
    xx = a1p[t][:, :, 0].T.dot(a1_loading.T)
    ax[0, 0].scatter(xx[:, 0], xx[:, 1], c=gR(i), s=s, edgecolor="none", lw=0)
    el = cplt.compute_ellipse(xx[:, 0], xx[:, 1])
    ax[0, 0].plot(el[0], el[1], c=gR(i), lw=2)
for i, t in enumerate(targets):
    xx = a1a[t][:, :, 0].T.dot(a1_loading.T)
    ax[0, 1].scatter(xx[:, 0], xx[:, 1], c=gR(i), s=s, edgecolor="none", lw=0)
    el = cplt.compute_ellipse(xx[:, 0], xx[:, 1])
    ax[0, 1].plot(el[0], el[1], c=gR(i), lw=2)
xmm = np.percentile(ax[0, 1].get_xlim()+ax[0, 0].get_xlim(), [0, 100])
ymm = np.percentile(ax[0, 1].get_ylim()+ax[0, 0].get_ylim(), [0, 100])
ax[0, 0].set_xlim(xmm); ax[0, 1].set_xlim(xmm)
ax[0, 0].set_ylim(ymm); ax[0, 1].set_ylim(ymm)

# PEG
catch = [c for c in pega.keys() if "CAT_" in c]
targets = catch+[t for t in pega.keys() if "TAR_" in t]
BwG, gR = thelp.make_tbp_colormaps(catch, targets, use_tar_freq_idx=0)
for i, t in enumerate(targets):
    xx = pegp[t][:, :, 0].T.dot(peg_loading.T)
    ax[1, 0].scatter(xx[:, 0], xx[:, 1], c=gR(i), s=s, edgecolor="none", lw=0)
    el = cplt.compute_ellipse(xx[:, 0], xx[:, 1])
    ax[1, 0].plot(el[0], el[1], c=gR(i), lw=2)
for i, t in enumerate(targets):
    xx = pega[t][:, :, 0].T.dot(peg_loading.T)
    ax[1, 1].scatter(xx[:, 0], xx[:, 1], c=gR(i), s=s, edgecolor="none", lw=0)
    el = cplt.compute_ellipse(xx[:, 0], xx[:, 1])
    ax[1, 1].plot(el[0], el[1], c=gR(i), lw=2)
xmm = np.percentile(ax[1, 1].get_xlim()+ax[1, 0].get_xlim(), [0, 100])
ymm = np.percentile(ax[1, 1].get_ylim()+ax[1, 0].get_ylim(), [0, 100])
ax[1, 0].set_xlim(xmm); ax[1, 1].set_xlim(xmm)
ax[1, 0].set_ylim(ymm); ax[1, 1].set_ylim(ymm)

for a in ax.flatten():
    a.set_xlabel(r"$\Delta \mu$")
    a.set_ylabel(r"$\sigma$")
    a.set_xticks([])
    a.set_yticks([])

f.patch.set_facecolor("white")
f.tight_layout()

f.savefig(os.path.join(figpath, "ellipse_plot_examples.svg"), dpi=500)