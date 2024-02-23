import nems0.db as nd
import numpy as np

batch = 324
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])
python_path = '/auto/users/hellerc/miniconda3/envs/lbhb/bin/python'
script = "/auto/users/hellerc/code/projects/TBP-ms/decoding/do_choice_decoding.py"
force_rerun = True
regress_pupil = False
shuffle = True

ndim = 2
# initial model list for testing
modellist = [
    ## Catch choice decoding
    f'tbpChoiceDecoding_decision.cr.ich_DRops.dim{ndim}.ddr',
    # cummulative time
    f'tbpChoiceDecoding_fs100_ws0.0_we0.05_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.1_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.15_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.2_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.25_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.3_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.35_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.4_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.45_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.5_decision.cr.ich_DRops.dim{ndim}.ddr',

    ## Target choice decoding
    f'tbpChoiceDecoding_decision.h.m_DRops.dim{ndim}.ddr',
    # cummulative time
    f'tbpChoiceDecoding_fs100_ws0.0_we0.05_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.1_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.15_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.2_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.25_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.3_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.35_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.4_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.45_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.5_decision.h.m_DRops.dim{ndim}.ddr',
]

if shuffle:
    modellist = [m+"_shuffle" for m in modellist]

if regress_pupil:
    modellist = [m+"_PR" for m in modellist]

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