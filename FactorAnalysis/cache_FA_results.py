"""
Inspired by Umakanthan 2021 neuron paper -- bridging neuronal correlations / dim reduction
Cache script --  
for each site, calculate:
    - % shared variance
    - loading similarity
    - dimensionality
    - also compute dimensionality / components of full space, including stimuli, using PCA (trial-averaged or raw data?)
        - think raw data is okay, but we'll do both. The point we want to make is that variability coflucates in a space with dim < total dim
"""
import numpy as np
from sklearn.decomposition import FactorAnalysis, PCA
from scipy.signal import argrelextrema
import charlieTools.TBP_ms.loaders as loaders
import pickle


import os
import sys
import nems0.db as nd
import logging 
import nems0

log = logging.getLogger(__name__)

if 'QUEUEID' in os.environ:
    queueid = os.environ['QUEUEID']
    nems0.utils.progress_fun = nd.update_job_tick

else:
    queueid = 0

if queueid:
    log.info("Starting QUEUEID={}".format(queueid))
    nd.update_job_start(queueid)


# Get sys args
site = sys.argv[1]  
batch = sys.argv[2]
modelname = sys.argv[3]

shuffle = False
perstim = False
for op in modelname.split("_"):
    if op == "shuff":
        shuffle = True
    if op == "perstim":
        perstim = True

# measure change in dimensionality, %sv, loading sim, across jackknifes
def get_dim(LL):
    try:
        return argrelextrema(LL, np.greater)[0][0]+1
    except:
        log.info("No relative LL max, choosing overall maximum")
        return np.argmax(LL)+1

def sigma_shared(model):
    return (model.components_.T @ model.components_)

def sigma_ind(model):
    return np.diag(model.noise_variance_)

def pred_cov(model):
    return sigma_shared(model) + sigma_ind(model)

def get_dim95(model):
    """
    number of dims to explain 95% of shared var
    """
    ss = sigma_shared(model)
    evals, _ = np.linalg.eig(ss)
    evals = evals / sum(evals)
    return np.argwhere(np.cumsum(evals)>=0.95)[0][0]+1

def get_sv(model):
    sig_shared = sigma_shared(model) # rank n_components cov matrix
    full_cov_pred = pred_cov(model)
    # % shared variance
    # per neuron
    pn = np.diag(sig_shared) / np.diag(full_cov_pred)
    # average
    sv = np.mean(pn)
    return sv

def get_loading_similarity(model, dim=0):
    # loading similarity
    loading = model.components_[dim, :]
    loading /= np.linalg.norm(loading)
    load_sim = 1 - (np.var(loading) / (1 / len(loading)))
    return load_sim


# ============ perform analysis ==================

# load data
X_active, _ = loaders.load_tbp_for_decoding(site=site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=["HIT_TRIAL", "MISS_TRIAL", "CORRECT_REJECT_TRIAL"],
                                    recache=False)
X_passive, _ = loaders.load_tbp_for_decoding(site=site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=["PASSIVE_EXPERIMENT"],
                                    recache=False)

# keep only CAT and TAR stims and turn into a 4D matrix (cells x reps x stim x tbins)
keep = [t for t in X_active.keys() if (("TAR_" in t) | ("CAT_" in t)) & (t in X_passive.keys())]
X_active = {k: v for k, v in X_active.items() if k in keep}
X_passive = {k: v for k, v in X_passive.items() if k in keep}

# FOR SETTING UP FITTING COMMON SPACE ACROSS ALL STIM
if perstim == False:
    # fit all stim together, after subtracting psth
    # "special" cross-validation -- fitting individual stims doesn't work, not enough data
    # instead, leave-one-stim out fitting to find dims that are shared / stimulus-independent
    nstim = len(keep)
    nCells = X_active[keep[0]].shape[0]
    X_asub = {k: v - v.mean(axis=1, keepdims=True) for k, v in X_active.items()}
    X_psub = {k: v - v.mean(axis=1, keepdims=True) for k, v in X_passive.items()}
    X_ata = {k: v.mean(axis=1, keepdims=True) for k, v in X_active.items()}
    X_pta = {k: v.mean(axis=1, keepdims=True) for k, v in X_passive.items()}

    if shuffle:
        raise NotImplementedError("Not implemented for active / passive shuffling, yet")

    nfold = nstim
    nComponents = 50
    if nCells < nComponents:
        nComponents = nCells

    log.info("\nComputing log-likelihood across models / nfolds")
    LL_active = np.zeros((nComponents, nfold))
    LL_passive = np.zeros((nComponents, nfold))
    for ii in np.arange(1, LL_active.shape[0]+1):
        log.info(f"{ii} / {LL_active.shape[0]}")
        fa = FactorAnalysis(n_components=ii, random_state=0) # init model
        for nf, kk in enumerate(keep):
            fit_keys = [x for x in keep if x != kk]

            fit_afa = np.concatenate([X_asub[k] for k in fit_keys], axis=1).squeeze()
            fit_pfa = np.concatenate([X_psub[k] for k in fit_keys], axis=1).squeeze()
            eval_afa = X_asub[kk].squeeze()
            eval_pfa = X_psub[kk].squeeze()

            # ACTIVE FACTOR ANALYSIS
            fa.fit(fit_afa.T) # fit model
            # Get LL score
            LL_active[ii-1, nf] = fa.score(eval_afa.T)

            # PASSIVE FACTOR ANALYSIS
            fa.fit(fit_pfa.T) # fit model
            # Get LL score
            LL_passive[ii-1, nf] = fa.score(eval_pfa.T)

    log.info("Estimating %sv and loading similarity for the 'best' model")
    # ACTIVE
    active_dim_sem = np.std([get_dim(LL_active[:, i]) for i in range(LL_active.shape[1])]) / np.sqrt(LL_active.shape[1])
    active_dim = get_dim(LL_active.mean(axis=-1))
    # fit the "best" model over jackknifes
    a_sv = np.zeros(nfold)
    a_loading_sim = np.zeros(nfold)
    a_dim95 = np.zeros(nfold)
    for nf, kk in enumerate(keep):
        fit_keys = [x for x in keep if x != kk]
        x = np.concatenate([X_asub[k] for k in fit_keys], axis=1).squeeze()
        fa_active = FactorAnalysis(n_components=active_dim, random_state=0) 
        fa_active.fit(x.T)
        a_sv[nf] = get_sv(fa_active)
        a_loading_sim[nf] = get_loading_similarity(fa_active)
        # get n dims needs to explain 95% of shared variance
        a_dim95[nf] = get_dim95(fa_active)

    # PASSIVE
    passive_dim_sem = np.std([get_dim(LL_passive[:, i]) for i in range(LL_passive.shape[1])]) / np.sqrt(LL_passive.shape[1])
    passive_dim = get_dim(LL_passive.mean(axis=-1))
    # fit the "best" model over jackknifes
    p_sv = np.zeros(nfold)
    p_loading_sim = np.zeros(nfold)
    p_dim95 = np.zeros(nfold)
    for nf, kk in enumerate(keep):
        fit_keys = [x for x in keep if x != kk]
        x = np.concatenate([X_psub[k] for k in fit_keys], axis=1).squeeze()
        fa_passive = FactorAnalysis(n_components=passive_dim, random_state=0) 
        fa_passive.fit(x.T)
        p_sv[nf] = get_sv(fa_passive)
        p_loading_sim[nf] = get_loading_similarity(fa_passive)
        # get n dims needs to explain 95% of shared variance
        p_dim95[nf] = get_dim95(fa_passive)


    # final fit with all data to get components
    fa_active = FactorAnalysis(n_components=active_dim, random_state=0) 
    x = np.concatenate([X_asub[k] for k in keep], axis=1).squeeze()
    fa_active.fit(x.T)
    active_sv_all = get_sv(fa_active)
    active_ls_all = get_loading_similarity(fa_active)
    active_dim95_all = get_dim95(fa_active)

    fa_passive = FactorAnalysis(n_components=passive_dim, random_state=0) 
    x = np.concatenate([X_psub[k] for k in keep], axis=1).squeeze()
    fa_passive.fit(x.T)
    passive_sv_all = get_sv(fa_passive)
    passive_ls_all = get_loading_similarity(fa_passive)
    passive_dim95_all = get_dim95(fa_passive)

    # Save results
    results = {
        "active_sv": a_sv.mean(),
        "passive_sv": p_sv.mean(),
        "active_sv_sd": a_sv.std(),
        "passive_sv_sd": p_sv.std(),
        "active_loading_sim": a_loading_sim.mean(),
        "passive_loading_sim": p_loading_sim.mean(),
        "active_loading_sim_sd": a_loading_sim.std(),
        "passive_loading_sim_sd": p_loading_sim.std(),
        "active_dim95": a_dim95.mean(),
        "passive_dim95": p_dim95.mean(),
        "active_dim95_sd": a_dim95.std(),
        "passive_dim95_sd": p_dim95.std(),
        "active_dim": active_dim,
        "passive_dim": passive_dim,
        "active_dim_sem": active_dim_sem,
        "passive_dim_sem": passive_dim_sem,
        "nCells": nCells,
        "nStim": nstim,
        "final_fit": {
            "fa_active.components_": fa_active.components_,
            "fa_passive.components_": fa_passive.components_,
            "fa_active.sigma_shared": sigma_shared(fa_active),
            "fa_passive.sigma_shared": sigma_shared(fa_passive),
            "fa_active.sigma_ind": sigma_ind(fa_active),
            "fa_passive.sigma_ind": sigma_ind(fa_passive),
            "fa_active.sigma_full": pred_cov(fa_active),
            "fa_passive.sigma_full": pred_cov(fa_passive),
            "active_sv_all": active_sv_all,
            "passive_sv_all": passive_sv_all,
            "active_ls_all": active_ls_all,
            "passive_ls_all": passive_ls_all,
            "active_dim95_all": active_dim95_all,
            "passive_dim95_all": passive_dim95_all
        }
    }


# FIT EACH STIM INDIVIDUALLY
else:
    # can't do CV here. Instead, just fit model for each stimulus
    nstim = len(keep)
    nCells = X_active[keep[0]].shape[0]
    X_asub = {k: v - v.mean(axis=1, keepdims=True) for k, v in X_active.items()}
    X_psub = {k: v - v.mean(axis=1, keepdims=True) for k, v in X_passive.items()}
    X_ata = {k: v.mean(axis=1, keepdims=True) for k, v in X_active.items()}
    X_pta = {k: v.mean(axis=1, keepdims=True) for k, v in X_passive.items()}

    if shuffle:
        raise NotImplementedError("Not implemented for active / passive shuffling, yet")

    nfold = nstim
    nComponents = 50
    if nCells < nComponents:
        nComponents = nCells

    log.info("\nComputing log-likelihood across models / stimuli")
    LL_active = np.zeros((nComponents, nstim))
    LL_passive = np.zeros((nComponents, nstim))
    for ii in np.arange(1, LL_active.shape[0]+1):
        log.info(f"{ii} / {LL_active.shape[0]}")
        fa = FactorAnalysis(n_components=ii, random_state=0) # init model
        for st, kk in enumerate(keep):
            # ACTIVE FACTOR ANALYSIS
            fa.fit(X_asub[kk].squeeze().T) # fit model
            # Get LL score
            LL_active[ii-1, st] = fa.score(X_asub[kk].squeeze().T)

            # PASSIVE FACTOR ANALYSIS
            fa.fit(X_psub[kk].squeeze().T) # fit model
            # Get LL score
            LL_passive[ii-1, st] = fa.score(X_psub[kk].squeeze().T)

    log.info("Estimating %sv and loading similarity for the 'best' model in each state")

    results = {"active": {}, "passive": {}}

    # ACTIVE
    active_dim = [get_dim(LL_active[:, i]) for i in range(LL_active.shape[1])]
    # fit the "best" model over jackknifes
    a_sv = np.zeros(nstim)
    a_loading_sim = np.zeros(nstim)
    a_dim95 = np.zeros(nstim)
    for st, kk in enumerate(keep):
        x = X_asub[kk].squeeze()
        fa_active = FactorAnalysis(n_components=active_dim[st], random_state=0) 
        fa_active.fit(x.T)
        a_sv[st] = get_sv(fa_active)
        a_loading_sim[st] = get_loading_similarity(fa_active)
        # get n dims needs to explain 95% of shared variance
        a_dim95[st] = get_dim95(fa_active)

        results["active"][kk] = {}
        results["active"][kk]["sv"] = a_sv
        results["active"][kk]["loading_sim"] = a_loading_sim
        results["active"][kk]["dim"] = a_dim95
        results["active"][kk]["components_"] = fa_active.components_
        results["active"][kk]["sigma_shared"] = sigma_shared(fa_active),
        results["active"][kk]["sigma_ind"] = sigma_ind(fa_active)
        results["active"][kk]["sigma_full"] = pred_cov(fa_active)

    # PASSIVE
    passive_dim = [get_dim(LL_passive[:, i]) for i in range(LL_passive.shape[1])]
    # fit the "best" model over jackknifes
    p_sv = np.zeros(nstim)
    p_loading_sim = np.zeros(nstim)
    p_dim95 = np.zeros(nstim)
    for st, kk in enumerate(keep):
        x = X_psub[kk].squeeze()
        fa_passive = FactorAnalysis(n_components=passive_dim[st], random_state=0) 
        fa_passive.fit(x.T)
        p_sv[st] = get_sv(fa_passive)
        p_loading_sim[st] = get_loading_similarity(fa_passive)
        # get n dims needs to explain 95% of shared variance
        p_dim95[st] = get_dim95(fa_passive)

        results["passive"][kk] = {}
        results["passive"][kk]["sv"] = p_sv
        results["passive"][kk]["loading_sim"] = p_loading_sim
        results["passive"][kk]["dim"] = p_dim95
        results["passive"][kk]["components_"] = fa_passive.components_
        results["passive"][kk]["sigma_shared"] = sigma_shared(fa_passive),
        results["passive"][kk]["sigma_ind"] = sigma_ind(fa_passive)
        results["passive"][kk]["sigma_full"] = pred_cov(fa_passive)


def save(d, path):
    with open(path+f'/{modelname}.pickle', 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None

path = "/auto/users/hellerc/results/TBP-ms/factor_analysis/"
if os.path.isdir(os.path.join(path, str(batch), site)):
   pass
elif os.path.isdir(os.path.join(path, str(batch))):
    os.mkdir(os.path.join(path, str(batch), site))
else:
    os.mkdir(os.path.join(path, str(batch)))
    os.mkdir(os.path.join(path, str(batch), site))

save(results, os.path.join(path, str(batch), site))

if queueid:
    nd.update_job_complete(queueid)