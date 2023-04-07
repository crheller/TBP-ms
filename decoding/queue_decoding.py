import nems0.db as nd
import numpy as np

batch = 324
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])
python_path = '/auto/users/hellerc/miniconda3/envs/lbhb/bin/python'
script = "/auto/users/hellerc/code/projects/TBP-ms/decoding/do_decoding.py"
force_rerun = True
FA_simulation = True
regress_pupil = True
ndim = 2
# initial model list for testing
modellist = [
    ## decoding axis specific to state (active vs. passive). This is default
    f'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim{ndim}.ddr-targetNoise',
    f'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim{ndim}.ddr-targetNoise',
    ## same as above, but match pupil size in passive to active, stricly
    #f'tbpDecoding_mask.paB_drmask.h.cr.m.pa_DRops.dim{ndim}.ddr-targetNoise',
    
    # no misses
    f'tbpDecoding_mask.h.cr_drmask.h.cr.pa_DRops.dim{ndim}.ddr-targetNoise',
    f'tbpDecoding_mask.pa_drmask.h.cr.pa_DRops.dim{ndim}.ddr-targetNoise',

    ## use only active mask for getting decoding axis
    # 'tbpDecoding_mask.h.cr.m_decmask.h.cr.m_drmask.h.cr.m.pa_DRops.dim{ndim}.ddr-targetNoise',
    # 'tbpDecoding_mask.pa_decmask.h.cr.m_drmask.h.cr.m.pa_DRops.dim{ndim}.ddr-targetNoise',
    
    ## use only passive mask for getting decoding axis
    # 'tbpDecoding_mask.h.cr.m_decmask.pa_drmask.h.cr.m.pa_DRops.dim{ndim}.ddr-targetNoise',
    # 'tbpDecoding_mask.pa_decmask.pa_drmask.h.cr.m.pa_DRops.dim{ndim}.ddr-targetNoise',

    ## use all data for getting decoding axis
    f'tbpDecoding_mask.h.cr.m_decmask.h.cr.m.pa_drmask.h.cr.m.pa_DRops.dim{ndim}.ddr-targetNoise',
    f'tbpDecoding_mask.pa_decmask.h.cr.m.pa_drmask.h.cr.m.pa_DRops.dim{ndim}.ddr-targetNoise',
    # no misses
    f'tbpDecoding_mask.h.cr_decmask.h.cr.pa_drmask.h.cr.pa_DRops.dim{ndim}.ddr-targetNoise',
    f'tbpDecoding_mask.pa_decmask.h.cr.pa_drmask.h.cr.pa_DRops.dim{ndim}.ddr-targetNoise',

    ## use all data for getting decoding axis, shared decoding space across stim pairs
    f'tbpDecoding_mask.h.cr.m_decmask.h.cr.m.pa_drmask.h.cr.m.pa_DRops.dim{ndim}.ddr-targetNoise-sharedSpace',
    f'tbpDecoding_mask.pa_decmask.h.cr.m.pa_drmask.h.cr.m.pa_DRops.dim{ndim}.ddr-targetNoise-sharedSpace',
    # no misses
    f'tbpDecoding_mask.h.cr_decmask.h.cr.pa_drmask.h.cr.pa_DRops.dim{ndim}.ddr-targetNoise-sharedSpace',
    f'tbpDecoding_mask.pa_decmask.h.cr.pa_drmask.h.cr.pa_DRops.dim{ndim}.ddr-targetNoise-sharedSpace',

    ## use state specific decoding axis, shared decoding space across stim pairs
    f'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim{ndim}.ddr-targetNoise-sharedSpace',
    f'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim{ndim}.ddr-targetNoise-sharedSpace',
    # no misses
    f'tbpDecoding_mask.h.cr_drmask.h.cr.pa_DRops.dim{ndim}.ddr-targetNoise-sharedSpace',
    f'tbpDecoding_mask.pa_drmask.h.cr.pa_DRops.dim{ndim}.ddr-targetNoise-sharedSpace',
]

if regress_pupil:
    modellist = [m+"_PR" for m in modellist]

if FA_simulation:
    # m1 = [m+"_FA.1" for m in modellist]
    # m2 = [m+"_FA.2" for m in modellist]
    # m3 = [m+"_FA.3" for m in modellist]
    # m4 = [m+"_FA.4" for m in modellist]
    
    # null model
    m0 = [m+"_FAperstim.0" for m in modellist]
    # no correlation models
    m1 = [m+"_FAperstim.5" for m in modellist]
    m2 = [m+"_FAperstim.6" for m in modellist]
    
    # "normal" models
    m5 = [m+"_FAperstim.1" for m in modellist]
    m6 = [m+"_FAperstim.2" for m in modellist]
    m7 = [m+"_FAperstim.3" for m in modellist]
    m8 = [m+"_FAperstim.4" for m in modellist]
    modellist = m0 + m1 + m2 + m5 + m6 + m7 + m8
    if regress_pupil:
        modellist = [m+".PR" for m in modellist]

modellist = [m for m in modellist if "FAperstim.0" in m]

bad_sites = [
    "ARM004e",
    "ARM005e",
    "CLT016a",
    "CRD013b"
]

sites = [s for s in sites if s not in bad_sites]

nd.enqueue_models(celllist=sites,
                        batch=batch,
                        modellist=modellist,
                        executable_path=python_path,
                        script_path=script,
                        user='hellerc',
                        force_rerun=force_rerun,
                        reserve_gb=2)