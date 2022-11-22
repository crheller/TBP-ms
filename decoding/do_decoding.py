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
    for mo in mask_ops:
        if mo=="h":
            mask.append("HIT_TRIAL")
        if mo=="cr":
            mask.append("CORRECT_REJECT_TRIAL")
        if mo=="m":
            mask.append("MISS_TRIAL")
        if mo=="fa":
            mask.append("FALSE_ALARM_TRIAL")
        if mo=="pa":
            mask.append("PASSIVE_EXPERIMENT")
    return mask

mask = []
drmask = []
decmask = []
method = "unknown"
ndims = 2
noise = "global"
sharedSpace = False
factorAnalysis = False
fa_perstim = False
sim = None
for op in modelname.split("_"):
    if op.startswith("mask"):
        mask = parse_mask_options(op)
    if op.startswith("drmask"):
        drmask = parse_mask_options(op)
    if op.startswith("decmask"):
        decmask = parse_mask_options(op)
    if op.startswith("DRops"):
        dim_reduction_options = op.split(".")
        for dro in dim_reduction_options:
            if dro.startswith("dim"):
                ndims = int(dro[3:])
            if dro.startswith("ddr"):
                method = "dDR"
                ddrops = dro.split("-")
                for ddr_op in ddrops:
                    if ddr_op == "globalNoise":
                        noise = "global"
                    elif ddr_op == "targetNoise":
                        noise = "targets"
                    elif ddr_op == "sharedSpace":
                        sharedSpace = True
    if op.startswith("FA"):
        factorAnalysis = True
        sim_method = int(op.split(".")[1])
        fa_perstim = op.split(".")[0][2:]=="perstim"

if decmask == []:
    # default is to compute decoding axis using the same data you're evaluating on
    decmask = mask

# STEP 3: Load data to be decoded / data to be use for decoding space definition
X, Xp = loaders.load_tbp_for_decoding(site=site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=mask,
                                    recache=False)

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
Xog = X.copy()
if factorAnalysis:
    # redefine X using simulated data
    if "PASSIVE_EXPERIMENT" in mask:
        state = "passive"
    else:
        state = "active"
    if fa_perstim:
        keep = [k for k in Xog.keys() if ("TAR_" in k) | ("CAT_" in k)]
        Xog = {k: v for k, v in Xog.items() if k in keep}
        psth = {k: v.mean(axis=1).squeeze() for k, v in Xog.items()}
        log.info("Loading FA simulation using per stimulus results")
        X = loaders.load_FA_model_perstim(site, batch, psth, state, sim=sim_method, nreps=2000)
        Xog = {k: v for k, v in Xog.items() if k in X.keys()}
    else:
        log.info("Loading FA simulation")
        psth = {k: v.mean(axis=1).squeeze() for k, v in X.items()}
        X = loaders.load_FA_model(site, batch, psth, state, sim=sim_method, nreps=2000)

# always define the space with the raw data, for the sake of comparison
Xd, _ = loaders.load_tbp_for_decoding(site=site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=drmask,
                                    balance=True)
Xdec, _ = loaders.load_tbp_for_decoding(site=site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=decmask,
                                    balance=True)

# STEP 4: Generate list of stimulus pairs meeting min rep criteria and get the decoding space for each
stim_pairs = list(combinations(Xog.keys(), 2))
stim_pairs = [sp for sp in stim_pairs if (Xog[sp[0]].shape[1]>=5) & (Xog[sp[1]].shape[1]>=5)]
# TODO: Add option to compute a single, fixed space for all pairs. e.g. a generic
# target vs. catch space.
decoding_space = decoding.get_decoding_space(Xd, stim_pairs, 
                                            method=method, 
                                            noise_space=noise,
                                            ndims=ndims,
                                            common_space=sharedSpace)

if len(decoding_space) != len(stim_pairs):
    raise ValueError

# STEP 4.1: Save a figure of projection of targets / catches a common decoding space for this site
fig_file = results_file(RESULTS_DIR, site, batch, modelname, "ellipse_plot.png")
plotting.dump_ellipse_plot(site, batch, filename=fig_file, mask=drmask)

# STEP 5: Loop over stimulus pairs and perform decoding
output = []
for sp, axes in zip(stim_pairs, decoding_space):
    # first, get decoding axis for this stim pair
    # TODO: Add specialty option for generic target vs. catch decoding space.
    _r1 = Xdec[sp[0]][:, :, 0]
    _r2 = Xdec[sp[1]][:, :, 0]
    _result = decoding.do_decoding(_r1, _r2, axes)
    
    r1 = X[sp[0]].squeeze()
    r2 = X[sp[1]].squeeze()
    result = decoding.do_decoding(r1, r2, axes, wopt=_result.wopt)
    pair_category = decoding.get_category(sp[0], sp[1])

    df = pd.DataFrame(index=["dp", "wopt", "evals", "evecs", "evecSim", "dU"],
                        data=[result.dprimeSquared,
                              result.wopt,
                              result.evals,
                              result.evecs,
                              result.evecSim,
                              result.dU]
                            ).T
    df["class"] = pair_category
    df["e1"] = sp[0]
    df["e2"] = sp[1]
    df["dr_loadings"] = [axes]

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
    'class': 'object',
    }
output = output.astype(dtypes)

# STEP 6: Save results
results_file = results_file(RESULTS_DIR, site, batch, modelname, "output.pickle")
output.to_pickle(results_file)

# STEP 7: Update queue monitor
if queueid:
    nd.update_job_complete(queueid)