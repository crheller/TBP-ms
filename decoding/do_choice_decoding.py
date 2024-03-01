"""
Decoding analysis. 
Dump dprime between stimulus pairs, calculated according to "model" options
    e.g. dDR over all day, on pairwise basis, exclude bad trials, cross-validation, etc.
"""

# STEP 1: Import modules and set up queue job
import charlieTools.TBP_ms.loaders as loaders
import charlieTools.TBP_ms.decoding as decoding
import charlieTools.TBP_ms.plotting as plotting
import pandas as pd
from itertools import combinations
import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms/")
from settings import RESULTS_DIR
from path_helpers import results_file
import os
import nems0
import nems0.db as nd
import logging
import numpy as np
np.random.seed(123)

log = logging.getLogger(__name__)

if 'QUEUEID' in os.environ:
    queueid = os.environ['QUEUEID']
    nems0.utils.progress_fun = nd.update_job_tick

else:
    queueid = 0

if queueid:
    log.info("Starting QUEUEID={}".format(queueid))
    nd.update_job_start(queueid)

# STEP 2: Read / parse system arguments
site = sys.argv[1]  
batch = int(sys.argv[2])
modelname = sys.argv[3]

def parse_mask_options(op):
    mask = []
    mask_ops = op.split(".")
    pup_match_active = False
    for mo in mask_ops:
        if mo=="h":
            mask.append("HIT_TRIAL")
        if mo=="cr":
            mask.append("CORRECT_REJECT_TRIAL")
        if mo=="ich":
            mask.append("INCORRECT_HIT_TRIAL")
        if mo=="m":
            mask.append("MISS_TRIAL")
        if mo=="fa":
            mask.append("FALSE_ALARM_TRIAL")
        if mo=="pa":
            mask.append("PASSIVE_EXPERIMENT")
        if mo=="paB":
            mask.append("PASSIVE_EXPERIMENT")
            pup_match_active = True
    return mask, pup_match_active

mask = []
method = "unknown"
ndims = 2
factorAnalysis = False
fa_perstim = False
sim = None
pup_match_active = False
regress_pupil = False
trial_epoch = False
fa_model = "FA_perstim_choice"
window_start = 0.1 # by default, use the full stimulus
window_end = 0.4 # by default, use the full stimulus
fs = 10
shuffle = False
for op in modelname.split("_"):
    if op.startswith("decision"):
        mask, pup_match_active = parse_mask_options(op)
    if op.startswith("DRops"):
        dim_reduction_options = op.split(".")
        for dro in dim_reduction_options:
            if dro.startswith("dim"):
                ndims = int(dro[3:])
            if dro.startswith("ddr"):
                method = "dDR"

    if op.startswith("PR"):
        regress_pupil = True
    if op.startswith("FA"):
        factorAnalysis = True
        sim_method = int(op.split(".")[1])
        fa_perstim = op.split(".")[0][2:]=="perstim"
        try:
            if op.split(".")[2]=="PR":
                fa_model = "FA_perstim_PR"
                log.info("Using pupil regressed FA models")
        except:
            log.info("Using raw (not pupil regressed) FA fit")
            pass
    
    if op.startswith("ws"):
        window_start = float(op[2:])
    if op.startswith("we"):
        window_end = float(op[2:])
    if op.startswith("fs"):
        fs = int(op[2:])
    if op.startswith("shuffle"):
        shuffle = True
    if op.startswith("trial"):
        trial_epoch = True

if len(mask) != 2:
    raise ValueError("decision mask should be len = 2. Condition 1 vs. condition 2 to be decoded (e.g. hit vs. miss)")

# STEP 3: Load data to be decoded
X0, _ = loaders.load_tbp_for_decoding(site=site, 
                                    batch=batch,
                                    fs=fs,
                                    wins=window_start,
                                    wine=window_end,
                                    collapse=True,
                                    mask=mask[0],
                                    recache=False,
                                    get_full_trials=trial_epoch,
                                    pupexclude=pup_match_active,
                                    regresspupil=regress_pupil)
X1, _ = loaders.load_tbp_for_decoding(site=site, 
                                    batch=batch,
                                    fs=fs,
                                    wins=window_start,
                                    wine=window_end,
                                    collapse=True,
                                    mask=mask[1],
                                    recache=False,
                                    get_full_trials=trial_epoch,
                                    pupexclude=pup_match_active,
                                    regresspupil=regress_pupil)

# sim:
#     0 = no change (null) model
#     1 = change in gain only
#     2 = change in indep only (fixing absolute covariance)
#     3 = change in indep only (fixing relative covariance - so off-diagonals change)
#     4 = change in everything (full FA simulation)
#     # extras:
#     5 = set off-diag to zero, only change single neuron var.
#     6 = set off-diag to zero, fix single neuorn var
#     7 = no change (and no correlations at all)
Xog0 = X0.copy()
Xog1 = X1.copy()
if factorAnalysis:    
    # raise ValueError("FA simulation for choice decoding is a WIP")
    # redefine X0 and X1 by simulating response data
    if fa_perstim:
        log.info(f"Loading factor analysis results from {fa_model}")
        if sim_method==0:
            log.info("Fixing PSTH between decisions to correct decision")
            if mask[0] == "HIT_TRIAL":
                keep = [k for k in X0.keys() if ("TAR_" in k)]
                X0 = {k: v for k, v in X0.items() if k in keep}
                psth0 = {k: v.mean(axis=1).squeeze() for k, v in X0.items()}
                # get X1 / psth1 (the same as X0 for sim==0)
                X1 = {k: v for k, v in X0.items() if k in keep}
                psth1 = {k: v.mean(axis=1).squeeze() for k, v in X0.items()}
            elif mask[0] == "CORRECT_REJECT_TRIAL":
                keep = [k for k in X0.keys() if ("CAT_" in k)]
                X0 = {k: v for k, v in X0.items() if k in keep}
                psth0 = {k: v.mean(axis=1).squeeze() for k, v in X0.items()}
                # get X1 / psth1 (the same as X0 for sim==0)
                X1 = {k: v for k, v in X0.items() if k in keep}
                psth1 = {k: v.mean(axis=1).squeeze() for k, v in X0.items()}
            else:
                raise ValueError(f"{mask[0]} cannot be the first trial type")
        else:
            log.info("Allow PSTH to change between decisions")
            # allow X / psth to change between the decision types
            if mask[0] == "HIT_TRIAL":
                keep = [k for k in X0.keys() if ("TAR_" in k)]
                X0 = {k: v for k, v in X0.items() if k in keep}
                psth0 = {k: v.mean(axis=1).squeeze() for k, v in X0.items()}
                X1 = {k: v for k, v in X1.items() if k in keep}
                psth1 = {k: v.mean(axis=1).squeeze() for k, v in X1.items()}
            elif mask[0] == "CORRECT_REJECT_TRIAL":
                keep = [k for k in X0.keys() if ("CAT_" in k)]
                X0 = {k: v for k, v in X0.items() if k in keep}
                psth0 = {k: v.mean(axis=1).squeeze() for k, v in X0.items()}
                X1 = {k: v for k, v in X1.items() if k in keep}
                psth1 = {k: v.mean(axis=1).squeeze() for k, v in X1.items()}
            else:
                raise ValueError(f"{mask[0]} cannot be the first trial type")

        log.info("Loading FA simulation using per stimulus results")
        X0 = loaders.load_choice_FA_model(site, batch, psth0, mask[0], fa_model=fa_model, sim=sim_method, nreps=2000)
        X1 = loaders.load_choice_FA_model(site, batch, psth1, mask[1], fa_model=fa_model, sim=sim_method, nreps=2000)
    
    else:
        raise ValueError("No 'overall' FA fit for choice decoding")

# always define the space with the raw, BALANCED data, for the sake of comparison
Xd0, _ = loaders.load_tbp_for_decoding(site=site, 
                                    batch=batch,
                                    fs=fs,
                                    wins=window_start,
                                    wine=window_end,
                                    collapse=True,
                                    mask=mask[0],
                                    get_full_trials=trial_epoch,
                                    balance_choice=True,
                                    regresspupil=regress_pupil)
Xd1, _ = loaders.load_tbp_for_decoding(site=site, 
                                    batch=batch,
                                    fs=fs,
                                    wins=window_start,
                                    wine=window_end,
                                    collapse=True,
                                    mask=mask[1],
                                    get_full_trials=trial_epoch,
                                    balance_choice=True,
                                    regresspupil=regress_pupil)

# STEP 4: Generate list of stimuli to calculate choice decoding of (need min reps in each condition)
# then, for each stimulus, define the dDR axes
poss_stim = list(set(Xd0.keys()) & set(Xd1.keys()))
all_stimuli = [s for s in poss_stim if (("TAR" in s) | ("CAT" in s)) & (Xd0[s].shape[1]>=5) & (Xd1[s].shape[1]>=5)]

if len(all_stimuli) == 0:
    raise ValueError("no stimuli matching requirements")

pairs = list(combinations(mask, 2))
decoding_space = []
for stim in all_stimuli:
    Xdecoding = dict.fromkeys(mask)
    Xdecoding[mask[0]] = Xd0[stim]
    Xdecoding[mask[1]] = Xd1[stim]
    decoding_space.append(decoding.get_decoding_space(Xdecoding, pairs, 
                                            method=method, 
                                            noise_space="global",
                                            ndims=ndims,
                                            common_space=False)[0])

if len(decoding_space) != len(all_stimuli):
    raise ValueError

# STEP 4.1: Save a figure of projection of targets / catches a common decoding space for this site
# fig_file = results_file(RESULTS_DIR, site, batch, modelname, "ellipse_plot.png")
# plotting.dump_ellipse_plot(site, batch, filename=fig_file, mask=drmask)

# STEP 5: Loop over stimuli and perform choice decoding
output = []
for sp, axes in zip(all_stimuli, decoding_space):
    # first, get decoding axis for this stim pair
    Xdecoding = dict.fromkeys(mask)    
    Xdecoding[mask[0]] = Xd0[sp]
    Xdecoding[mask[1]] = Xd1[sp]

    _r1 = Xdecoding[mask[0]][:, :, 0]
    _r2 = Xdecoding[mask[1]][:, :, 0]
    _result = decoding.do_decoding(_r1, _r2, axes)
    
    # then do decoding on this axis (with the potentially unbalanced data)
    X = dict.fromkeys(mask)
    if shuffle:
        # if shuffling, run 1000 times to get mean across shuffles
        # keep mean dprime and mean percent correct.
        # for the rest of "results", just save the last shuffle result
        pc = []
        dprime = []
        for sidx in range(1000):
            all_reps = np.concatenate([X0[sp], X1[sp]], axis=1)
            nreps = all_reps.shape[1]
            rep1 = X0[sp].shape[1]
            take1 = np.random.choice(np.arange(nreps), rep1, replace=False)
            take2 = np.array(list(set(np.arange(nreps)).difference(take1)))
            X[mask[0]] = all_reps[:, take1, :]
            X[mask[1]] = all_reps[:, take2, :]

            r1 = X[mask[0]].squeeze()
            r2 = X[mask[1]].squeeze()
            result = decoding.do_decoding(r1, r2, axes, wopt=_result.wopt)

            # project r1 and r2 onto the optimal decoding axis, find decision boundary, count %correct labeled
            # very important to do cross validation here, otherwise percent correct calc is circular
            r1proj = r1.T @ axes.T @ _result.wopt
            r2proj = r2.T @ axes.T @ _result.wopt
            # leave-one-out cross validation 
            all_data = np.concatenate([r1proj, r2proj])
            labels = np.concatenate([np.zeros((r1proj.shape[0])), np.ones((r2proj.shape[0]))])
            pred_lab = np.zeros((labels.shape[0]))
            for i in range(r1proj.shape[0]+r2proj.shape[0]):
                fit = np.array(list(set(np.arange(all_data.shape[0])).difference(set([i]))))
                _labels = labels[fit]
                _data = all_data[fit]
                r1u = np.mean(_data[_labels==0])
                r2u = np.mean(_data[_labels==1])
                if (abs(all_data[i]-r1u)) < (abs(all_data[i]-r2u)):
                    pred_lab[i] = 0
                else:
                    pred_lab[i] = 1
            percent_correct = np.mean(labels==pred_lab)
            dprime.append(result.dprimeSquared)
            pc.append(percent_correct)
        
        percent_correct = np.mean(pc)
        result._replace(dprimeSquared = np.mean(dprime))

    else:
        # if not shuffling, just run this one time
        X[mask[0]] = X0[sp]
        X[mask[1]] = X1[sp]
    
        r1 = X[mask[0]].squeeze()
        r2 = X[mask[1]].squeeze()
        result = decoding.do_decoding(r1, r2, axes, wopt=_result.wopt)

        # project r1 and r2 onto the optimal decoding axis, find decision boundary, count %correct labeled
        # very important to do cross validation here, otherwise percent correct calc is circular
        r1proj = r1.T @ axes.T @ _result.wopt
        r2proj = r2.T @ axes.T @ _result.wopt
        # leave-one-out cross validation 
        all_data = np.concatenate([r1proj, r2proj])
        labels = np.concatenate([np.zeros((r1proj.shape[0])), np.ones((r2proj.shape[0]))])
        pred_lab = np.zeros((labels.shape[0]))
        for i in range(r1proj.shape[0]+r2proj.shape[0]):
            fit = np.array(list(set(np.arange(all_data.shape[0])).difference(set([i]))))
            _labels = labels[fit]
            _data = all_data[fit]
            r1u = np.mean(_data[_labels==0])
            r2u = np.mean(_data[_labels==1])
            if (abs(all_data[i]-r1u)) < (abs(all_data[i]-r2u)):
                pred_lab[i] = 0
            else:
                pred_lab[i] = 1
        percent_correct = np.mean(labels==pred_lab)


    df = pd.DataFrame(index=["dp", "percent_correct", "wopt", "evals", "evecs", "evecSim", "dU"],
                        data=[result.dprimeSquared,
                              percent_correct,
                              _result.wopt,
                              result.evals,
                              result.evecs,
                              result.evecSim,
                              result.dU]
                            ).T
    df["e1"] = mask[0]
    df["e2"] = mask[1]
    df["dr_loadings"] = [axes]
    df["stimulus"] = sp

    output.append(df)

output = pd.concat(output)

dtypes = {
    'dp': 'float32',
    'percent_correct': 'float32',
    'wopt': 'object',
    'evecs': 'object',
    'evals': 'object',
    'evecSim': 'float32',
    'dU': 'object',
    'e1': 'object',
    'e2': 'object',
    "dr_loadings": "object",
    "stimulus": "object"
    }
output = output.astype(dtypes)

# STEP 6: Save results
results_file = results_file(RESULTS_DIR, site, batch, modelname, "output.pickle")
output.to_pickle(results_file)

# STEP 7: Update queue monitor
if queueid:
    nd.update_job_complete(queueid)