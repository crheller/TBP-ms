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
fa_model = "FA_perstim"
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
            log.info("Using pupil regressed FA models")
            if op.split(".")[2]=="PR":
                fa_model = "FA_perstim_PR"
        except:
            log.info("Didn't find a pupil regress FA option")
            pass

if len(mask) != 2:
    raise ValueError("decision mask should be len = 2. Condition 1 vs. condition 2 to be decoded (e.g. hit vs. miss)")

# STEP 3: Load data to be decoded / data to be use for decoding space definition
X0, _ = loaders.load_tbp_for_decoding(site=site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=mask[0],
                                    recache=False,
                                    pupexclude=pup_match_active,
                                    regresspupil=regress_pupil)
X1, _ = loaders.load_tbp_for_decoding(site=site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=mask[1],
                                    recache=False,
                                    pupexclude=pup_match_active,
                                    regresspupil=regress_pupil)

# for null simulation, load the mean PSTH regardless of current state
X_all, _ = loaders.load_tbp_for_decoding(site=site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=["HIT_TRIAL", "CORRECT_REJECT_TRIAL", "MISS_TRIAL", "FALSE_ALARM_TRIAL"],
                                    recache=False,
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
# Xog = X.copy()
if factorAnalysis:
    raise NotImplementedError("Haven't implemented FA for choice decoding yet")
    # redefine X using simulated data
    if "PASSIVE_EXPERIMENT" in mask:
        state = "passive"
    else:
        state = "active"
    if fa_perstim:
        log.info(f"Loading factor analysis results from {fa_model}")
        if sim_method==0:
            log.info("Fixing PSTH between active / passive to active")
            keep = [k for k in X_all.keys() if ("TAR_" in k) | ("CAT_" in k)]
            X_all = {k: v for k, v in X_all.items() if k in keep}
            psth = {k: v.mean(axis=1).squeeze() for k, v in X_all.items()}
            Xog = {k: v for k, v in X_all.items() if k in X.keys()}
        else:
            keep = [k for k in Xog.keys() if ("TAR_" in k) | ("CAT_" in k)]
            Xog = {k: v for k, v in Xog.items() if k in keep}
            psth = {k: v.mean(axis=1).squeeze() for k, v in Xog.items()}
            Xog = {k: v for k, v in Xog.items() if k in X.keys()}

        log.info("Loading FA simulation using per stimulus results")
        X = loaders.load_FA_model_perstim(site, batch, psth, state, fa_model=fa_model, sim=sim_method, nreps=2000)
    else:
        log.info("Loading FA simulation")
        psth = {k: v.mean(axis=1).squeeze() for k, v in X.items()}
        X = loaders.load_FA_model(site, batch, psth, state, sim=sim_method, nreps=2000)

# always define the space with the raw, BALANCED data, for the sake of comparison
Xd0, _ = loaders.load_tbp_for_decoding(site=site, 
                                    batch=batch,
                                    wins=0.1,
                                    wine=0.4,
                                    collapse=True,
                                    mask=mask[0],
                                    balance_choice=True,
                                    regresspupil=regress_pupil)
Xd1, _ = loaders.load_tbp_for_decoding(site=site, 
                                    batch=batch,
                                    wins=0.1,
                                    wine=0.4,
                                    collapse=True,
                                    mask=mask[1],
                                    balance_choice=True,
                                    regresspupil=regress_pupil)

# STEP 4: Generate list of stimuli to calculate choice decoding of (need min reps in each condition)
# then, for each stimulus, define the dDR axes
all_stimuli = [s for s in Xd0.keys() if (s.startswith("TAR") | s.startswith("CAT")) & (Xd0[s].shape[1]>=5)]

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
    
    X = dict.fromkeys(mask)
    X[mask[0]] = X0[sp]
    X[mask[1]] = X1[sp]
    r1 = X[mask[0]].squeeze()
    r2 = X[mask[1]].squeeze()
    result = decoding.do_decoding(r1, r2, axes, wopt=_result.wopt)

    df = pd.DataFrame(index=["dp", "wopt", "evals", "evecs", "evecSim", "dU"],
                        data=[result.dprimeSquared,
                              result.wopt,
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