"""
For each site / sound, cache a modulation index (active - passive / active + passive)
for each cell *with a significant response*
"""
import nems0.db as nd
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems_lbhb.tin_helpers as thelp
import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from settings import RESULTS_DIR, BAD_SITES

import os
import numpy as np
import pandas as pd
import scipy.stats as stats

savepath = os.path.join(RESULTS_DIR, "modulation_index.csv")

batch = 324
fs = 10
sites = [s for s in nd.get_batch_sites(batch)[0] if s not in BAD_SITES]
amask = ["HIT_TRIAL", "CORRECT_REJECT_TRIAL"]
pmask = ["PASSIVE_EXPERIMENT"]
options = {'resp': True, 'pupil': True, 'rasterfs': fs, 'stim': False}

cols = ["MI", "epoch", "category", "snr", "site", "area"]
ss = int(0.1 * fs)
ee = int(0.4 * fs)
dfs = []
for site in sites:
    manager = BAPHYExperiment(batch=batch, cellid=site, rawid=None)
    rec = manager.get_recording(recache=False, **options)
    rec['resp'] = rec['resp'].rasterize()
    rec = rec.create_mask(True)
    arec = rec.and_mask(amask)
    prec = rec.and_mask(pmask)

    ref, tars, _ = thelp.get_sound_labels(rec)
    snrs = thelp.get_snrs(tars)
    mi = []
    sound_id = []
    category = []
    snr_string = []
    for t, snr in zip(tars, snrs):
        try:
            atresp = rec["resp"].extract_epoch(t, mask=arec["mask"])
            ptresp = rec["resp"].extract_epoch(t, mask=prec["mask"])
        except IndexError:
            print(f"{t} did not exist in both active and passive")
            continue
        if (atresp.shape[0] > 5) & (ptresp.shape[0] > 5):
            at_psth = atresp.mean(axis=0)[:, ss:ee].sum(axis=-1)
            pt_psth = ptresp.mean(axis=0)[:, ss:ee].sum(axis=-1)

            # remove cells that don't have a significant response
            kk = np.zeros(atresp.shape[1]).astype(bool)
            for ii in range(atresp.shape[1]):
                ap = stats.ranksums(atresp[:, ii, :ss].flatten(), atresp[:, ii, ss:ee].flatten()).pvalue
                pp = stats.ranksums(ptresp[:, ii, :ss].flatten(), ptresp[:, ii, ss:ee].flatten()).pvalue
                if (ap < 0.05) | (pp < 0.05):
                    kk[ii] = True

            at_psth = at_psth[kk]
            pt_psth = pt_psth[kk]

            _mi = (at_psth - pt_psth) / (at_psth + pt_psth)
            cat = "catch" if "CAT_" in t else "target"
            mi.extend(_mi)
            sound_id.extend([t]*len(_mi))
            category.extend([cat]*len(_mi))
            snr_string.extend([float(snr)]*len(_mi))
    
    area = area = nd.pd_query(sql="SELECT area from sCellFile where cellid like %s", params=(f"%{site}%",)).iloc[0][0]
    site_id = [site] * len(mi)
    area = [area] * len(mi)
    data = np.stack([mi, sound_id, category, snr_string, site_id, area])
    dfs.append(pd.DataFrame(columns=cols, data=data.T))

df = pd.concat(dfs)
df = df.astype({
    "MI": float,
    "area": object,
    "category": object,
    "snr": float,
    "epoch": object,
    "site": object
})

df.to_csv(savepath)