import nems0.db as nd
import numpy as np

batch = 324
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])
python_path = '/auto/users/hellerc/miniconda3/envs/lbhb/bin/python'
script = "/auto/users/hellerc/code/projects/TBP-ms/decoding/do_decoding.py"
force_rerun = False
FA_simulation = True

# initial model list for testing
modellist = [
    ## decoding axis specific to state (active vs. passive). This is default
    'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',
    'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',

    ## use only active mask for getting decoding axis
    # 'tbpDecoding_mask.h.cr.m_decmask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',
    # 'tbpDecoding_mask.pa_decmask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',
    
    ## use only passive mask for getting decoding axis
    # 'tbpDecoding_mask.h.cr.m_decmask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',
    # 'tbpDecoding_mask.pa_decmask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',

    ## use all data for getting decoding axis
    'tbpDecoding_mask.h.cr.m_decmask.h.cr.m.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',
    'tbpDecoding_mask.pa_decmask.h.cr.m.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',

    ## use all data for getting decoding axis, shared decoding space across stim pairs
    'tbpDecoding_mask.h.cr.m_decmask.h.cr.m.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise-sharedSpace',
    'tbpDecoding_mask.pa_decmask.h.cr.m.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise-sharedSpace',

    ## use state specific decoding axis, shared decoding space across stim pairs
    'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise-sharedSpace',
    'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise-sharedSpace',
]

if FA_simulation:
    m1 = [m+"_FA.1" for m in modellist]
    m2 = [m+"_FA.2" for m in modellist]
    m3 = [m+"_FA.3" for m in modellist]
    m4 = [m+"_FA.4" for m in modellist]
    modellist = m1 + m2 + m3 + m4

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