import nems0.db as nd
import numpy as np

batch = 324
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])
python_path = '/auto/users/hellerc/anaconda3/envs/lbhb/bin/python'
script = "/auto/users/hellerc/code/projects/TBP-ms/decoding/do_decoding.py"
force_rerun = True

# initial model list for testing
modellist = [
    # decoding axis specific to state (active vs. passive)
    # 'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',
    # 'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',

    # don't include misses
    'tbpDecoding_mask.h.cr_drmask.h.cr.pa_DRops.dim2.ddr-targetNoise',
    'tbpDecoding_mask.pa_drmask.h.cr.pa_DRops.dim2.ddr-targetNoise',

    # active mask for getting decoding axis
    # 'tbpDecoding_mask.h.cr.m_decmask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',
    # 'tbpDecoding_mask.pa_decmask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',
    
    # passive mask for getting decoding axis
    # 'tbpDecoding_mask.h.cr.m_decmask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',
    # 'tbpDecoding_mask.pa_decmask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',
]

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