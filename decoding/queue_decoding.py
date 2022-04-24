import nems.db as nd
import numpy as np

batch = 324
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])
python_path = '/auto/users/hellerc/anaconda3/envs/lbhb/bin/python'
script = "/auto/users/hellerc/code/projects/TBP-ms/decoding/do_decoding.py"
force_rerun = True

# initial model list for testing
# need to add options (e.g. choice of data for decoding axis calc.)
modellist = [
    'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',
    'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',
]

nd.enqueue_models(celllist=sites,
                        batch=batch,
                        modellist=modellist,
                        executable_path=python_path,
                        script_path=script,
                        user='hellerc',
                        force_rerun=force_rerun,
                        reserve_gb=2)