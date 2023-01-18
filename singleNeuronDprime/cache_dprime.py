"""
Quick analysis to calculate d-prime between each target and catch
for each neuron. Save in one big dataframe.
"""
from itertools import combinations
from nems_lbhb.baphy_experiment import BAPHYExperiment
from dDR.utils.decoding import compute_dprime
import nems_lbhb.tin_helpers as thelp
import scipy.ndimage.filters as sf
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

min_trials = 5
batch = 324
sites = [s for s in nd.get_batch_sites(batch)[0] if s not in BAD_SITES]
fs = 10
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

    cc = list(combinations(tars, 2))

    for c in cc:
        try:
            ar1 = arec["resp"].extract_epoch(c[0], mask=arec["mask"])[:, :, 1:4].mean(axis=-1)
            ar2 = arec["resp"].extract_epoch(c[1], mask=arec["mask"])[:, :, 1:4].mean(axis=-1)
            pr1 = prec["resp"].extract_epoch(c[0], mask=prec["mask"])[:, :, 1:4].mean(axis=-1)
            pr2 = prec["resp"].extract_epoch(c[1], mask=prec["mask"])[:, :, 1:4].mean(axis=-1)

            # check that all responses meet minimum trial criteria
            if (ar1.shape[1]>=min_trials) & (ar2.shape[1]>=min_trials) & (pr1.shape[1]>=min_trials) & (pr2.shape[1]>=min_trials):
                # compute dprime for each neuron 
                dp = np.zeros((ar1.shape[1], 2))
                cellid = []
                e1 = []
                e2 = []
                cat = []
                for n in range(ar1.shape[1]):
                    adprime = np.sqrt(abs(compute_dprime(ar1[:, [n]].T, ar2[:, [n]].T)))
                    pdprime = np.sqrt(abs(compute_dprime(pr1[:, [n]].T, pr2[:, [n]].T)))
                    cid = arec["resp"].chans[n]

                    if ("TAR" in c[0]) & ("TAR" in c[1]):
                        category = "tar_tar"
                    elif ("CAT" in c[0]) & ("CAT" in c[1]):
                        category = "cat_cat"
                    else:
                        category = "tar_cat"

                    dp[n, :] = [adprime, pdprime]
                    cellid.append(cid)
                    e1.append(c[0])
                    e2.append(c[1])
                    cat.append(category)
                df = pd.DataFrame(data=dp, columns=["active", "passive"])
                df["cellid"]=cellid; df["e1"]=e1; df["e2"]=e2; df["category"]=cat;df["area"]=area
                df["site"] = site
                dfs.append(df)
        except:
            print(f"{c} didn't have matching epochs between passive/active")

df = pd.concat(dfs)
df.to_csv("/auto/users/hellerc/results/TBP-ms/singleNeuronDprime.csv")