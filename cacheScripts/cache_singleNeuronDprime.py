"""
Quick analysis to calculate d-prime between each target and catch
for each neuron. Save in one big dataframe.
"""
from itertools import combinations
from nems_lbhb.baphy_experiment import BAPHYExperiment
from dDR.utils.decoding import compute_dprime
import nems_lbhb.tin_helpers as thelp
import scipy.stats as ss
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
n_resamples = 100
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
                null_low_ci = []
                null_high_ci = []
                null_ahigh_ci = []
                null_phigh_ci = []
                for n in range(ar1.shape[1]):
                    adprime = np.sqrt(abs(compute_dprime(ar1[:, [n]].T, ar2[:, [n]].T)))
                    pdprime = np.sqrt(abs(compute_dprime(pr1[:, [n]].T, pr2[:, [n]].T)))
                    cid = arec["resp"].chans[n]

                    # resample to get a "null" distribution of active / passive dprimes
                    # basic idea is, randomly assign a label to each response (active / passive)
                    # then recompute dprime to define the null distribution for that stimulus pair
                    ndraw1 = np.min([ar1.shape[0], pr1.shape[0]])
                    ndraw2 = np.min([ar2.shape[0], pr2.shape[0]])
                    adprime_null = np.zeros(n_resamples)
                    pdprime_null = np.zeros(n_resamples)
                    for rs in range(n_resamples):
                        ss1_all = np.concatenate(
                                [
                                    ar1[np.random.choice(np.arange(ar1.shape[0]), ndraw1, replace=True), [n]],
                                    pr1[np.random.choice(np.arange(pr1.shape[0]), ndraw1, replace=True), [n]]
                                ], axis=0
                        )
                        ss2_all = np.concatenate(
                                [
                                    ar2[np.random.choice(np.arange(ar2.shape[0]), ndraw2, replace=True), [n]],
                                    pr2[np.random.choice(np.arange(pr2.shape[0]), ndraw2, replace=True), [n]]
                                ], axis=0
                        )
                        np.random.shuffle(ss1_all); np.random.shuffle(ss2_all)
                        adprime_null[rs] = np.sqrt(abs(compute_dprime(ss1_all[:ndraw1, np.newaxis].T, ss2_all[:ndraw2, np.newaxis].T)))
                        pdprime_null[rs] = np.sqrt(abs(compute_dprime(ss1_all[ndraw1:, np.newaxis].T, ss2_all[ndraw2:, np.newaxis].T)))

                    null_delta_low_ci = np.quantile(adprime_null-pdprime_null, 0.025)
                    null_delta_high_ci = np.quantile(adprime_null-pdprime_null, 0.975)

                    # null_active_high = np.quantile(adprime_null, 0.95)
                    # null_passive_high = np.quantile(pdprime_null, 0.95)

                    # second resampling to test if d-prime itself for this pair is significant
                    # idea here is to randomly assign a stimulus ID to each repetition and
                    # compute a null distribution of dprime. Do it separately for active / passive
                    ndraw_active = np.min([ar1.shape[0], ar2.shape[0]])
                    ndraw_passive = np.min([pr1.shape[0], pr2.shape[0]])
                    adprime_null = np.zeros(n_resamples)
                    pdprime_null = np.zeros(n_resamples)
                    for rs in range(n_resamples):
                        ssActive_all = np.concatenate(
                                [
                                    ar1[np.random.choice(np.arange(ar1.shape[0]), ndraw_active, replace=True), [n]],
                                    ar2[np.random.choice(np.arange(ar2.shape[0]), ndraw_active, replace=True), [n]]
                                ], axis=0
                        )
                        ssPassive_all = np.concatenate(
                                [
                                    pr1[np.random.choice(np.arange(pr1.shape[0]), ndraw_passive, replace=True), [n]],
                                    pr2[np.random.choice(np.arange(pr2.shape[0]), ndraw_passive, replace=True), [n]]
                                ], axis=0
                        )
                        np.random.shuffle(ssActive_all); np.random.shuffle(ssPassive_all)
                        adprime_null[rs] = np.sqrt(abs(compute_dprime(ssActive_all[:ndraw_active, np.newaxis].T, ssActive_all[ndraw_active:, np.newaxis].T)))
                        pdprime_null[rs] = np.sqrt(abs(compute_dprime(ssPassive_all[:ndraw_passive, np.newaxis].T, ssPassive_all[ndraw_passive:, np.newaxis].T)))

                    null_active_high = np.quantile(adprime_null, 0.95)
                    null_passive_high = np.quantile(pdprime_null, 0.95)

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
                    null_low_ci.append(null_delta_low_ci)
                    null_high_ci.append(null_delta_high_ci)
                    null_ahigh_ci.append(null_active_high)
                    null_phigh_ci.append(null_passive_high)

                df = pd.DataFrame(data=dp, columns=["active", "passive"])
                df["cellid"]=cellid; df["e1"]=e1; df["e2"]=e2; df["category"]=cat;df["area"]=area
                df["null_delta_low_ci"] = null_low_ci
                df["null_delta_high_ci"] = null_high_ci
                df["null_active_high_ci"] = null_ahigh_ci
                df["null_passive_high_ci"] = null_phigh_ci
                df["site"] = site
                dfs.append(df)
        except:
            print(f"{c} didn't have matching epochs between passive/active")

df = pd.concat(dfs)
df.to_csv("/auto/users/hellerc/results/TBP-ms/singleNeuronDprime.csv")