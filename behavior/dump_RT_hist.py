"""
RT histograms for each site
"""
import statistics
import nems_lbhb.tin_helpers as thelp
import pickle
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.behavior import get_reaction_times
from nems_lbhb.behavior_plots import plot_RT_histogram
import nems0.db as nd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 12

import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from settings import BAD_SITES

figpath = "/auto/users/hellerc/code/projects/TBP-ms/temp_figs/behavior/"
sites = nd.get_batch_sites(324)[0]
sites = [s for s in sites if s not in BAD_SITES]

for site in sites:
    manager = BAPHYExperiment(cellid=site, batch=324)
    exptparams = manager.get_baphy_exptparams()
    # get just active files
    parmfiles = [f for i, f in enumerate(manager.parmfile) if exptparams[i]["BehaveObjectClass"]!="Passive"]
    manager = BAPHYExperiment(parmfiles)
    options = {
                "resp": False, 
                "pupil": False, 
                "rasterfs": 20, 
                "keep_following_incorrect_trial": True, 
                "keep_cue_trials": True, 
                "keep_early_trials": True
        }
    performance = manager.get_behavior_performance(**options)

    bev = manager.get_behavior_events(**options)
    bev = manager._stack_events(bev)
    bev = bev[bev.invalidTrial==False]
    _rts = get_reaction_times(manager.get_baphy_exptparams()[0], bev, **options)

    targets = list(_rts['Target'].keys())
    tarsort = np.argsort([t.split("+")[1].strip("dB") for t in targets])
    targets = np.array(targets)[tarsort]
    main_freq = statistics.mode(thelp.get_freqs(targets))
    cat = [t for t in targets if ('-Inf' in t) & (str(main_freq) in t)][0]
    targets = [t for t in targets if (str(main_freq) in t) & (t != cat)]
    targets = [cat] + targets
    # only keep targets with freq. matching the catch
    snrs = thelp.get_snrs(targets)
    rts = dict.fromkeys([str(s) for s in snrs])
    DI = dict.fromkeys([str(s) for s in snrs])
    count = dict.fromkeys([str(s) for s in snrs])
    for s, t in zip(snrs, targets):
        rts[str(s)] = _rts['Target'][t]
        count[str(s)] = len(_rts["Target"][t])
        _t = t.split(':')[0]
        if '-Inf' not in _t:
            try:
                DI[str(s)] = performance['dprime'][_t+'_'+cat.split(':')[0]]
            except:
                DI[str(s)] = performance['dprime'][_t+'+reminder_'+cat.split(':')[0]]

    BwG, gR = thelp.make_tbp_colormaps([cat], targets, use_tar_freq_idx=0)
    legend = [s+ f" dB, n={count[s]}, d'={round(DI[s], 3)}" if '-inf' not in s else 'Catch' for s in rts.keys()]

    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    bins = np.arange(0, 1.4, 0.001)
    plot_RT_histogram(rts, bins=bins, ax=ax, cmap=gR, lw=2, legend=legend)
    ax.set_title(site)
    f.tight_layout()
    f.savefig(os.path.join(figpath, f"{site}_RTs"))