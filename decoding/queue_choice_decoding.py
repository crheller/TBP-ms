import nems0.db as nd
import numpy as np

batch = 324
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])
python_path = '/auto/users/hellerc/miniconda3/envs/lbhb/bin/python'
script = "/auto/users/hellerc/code/projects/TBP-ms/decoding/do_choice_decoding.py"
force_rerun = True
regress_pupil = False
shuffle = False
FA = False
models = "targets" # "catch" "targets"

bad_sites = [
    "ARM004e",
    "ARM005e",
    "CLT016a",
    "CRD013b",
    # additional list of bad sites for choice decoding (determined not to have valid stimuli)
]
bad_catch_sites = [
    "ARM007c",
    "CLT017a",
    "CLT021a",
    "CLT023a",
    "CRD012b",
    "JLY007d"
]
bad_target_sites = [
    "CLT009a",
    "CRD019b",
    "JLY008b",
    "JLY010b",
    "JLY014d"
]

ndim = 2
# sliding window through the target/catch stimulus
# catch_modellist = [
#     ## Catch choice decoding
#     f'tbpChoiceDecoding_decision.cr.ich_DRops.dim{ndim}.ddr',
#     # sliding window
#     f'tbpChoiceDecoding_fs100_ws0.0_we0.1_decision.cr.ich_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.05_we0.15_decision.cr.ich_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.1_we0.2_decision.cr.ich_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.15_we0.25_decision.cr.ich_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.2_we0.3_decision.cr.ich_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.25_we0.35_decision.cr.ich_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.3_we0.4_decision.cr.ich_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.35_we0.45_decision.cr.ich_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.4_we0.5_decision.cr.ich_DRops.dim{ndim}.ddr',
# ]

# target_modellist = [
#     ## Target choice decoding
#     f'tbpChoiceDecoding_decision.h.m_DRops.dim{ndim}.ddr',
#     # sliding window
#     f'tbpChoiceDecoding_fs100_ws0.0_we0.1_decision.h.m_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.05_we0.15_decision.h.m_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.1_we0.2_decision.h.m_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.15_we0.25_decision.h.m_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.2_we0.3_decision.h.m_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.25_we0.35_decision.h.m_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.3_we0.4_decision.h.m_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.35_we0.45_decision.h.m_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.4_we0.5_decision.h.m_DRops.dim{ndim}.ddr',
# ]

# sliding window across entire trial
# t = 0.0 is aligned to (target/catch) sound onset
# so ws0.0_we0.1_trial should be equivalent to ws0.0_we0.1 etc.
catch_modellist = [
    ## Catch choice decoding
    f'tbpChoiceDecoding_fs100_ws-0.5_we-0.4_trial_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws-0.4_we-0.3_trial_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws-0.3_we-0.2_trial_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws-0.2_we-0.1_trial_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws-0.1_we-0.0_trial_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.1_trial_decision.cr.ich_DRops.dim{ndim}.ddr', # sound onset
    f'tbpChoiceDecoding_fs100_ws0.1_we0.2_trial_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.2_we0.3_trial_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.3_we0.4_trial_decision.cr.ich_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.4_we0.5_trial_decision.cr.ich_DRops.dim{ndim}.ddr',
]
target_modellist = [
    ## Target choice decoding
    f'tbpChoiceDecoding_fs100_ws-0.5_we-0.4_trial_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws-0.4_we-0.3_trial_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws-0.3_we-0.2_trial_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws-0.2_we-0.1_trial_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws-0.1_we-0.0_trial_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.0_we0.1_trial_decision.h.m_DRops.dim{ndim}.ddr', # sound onset
    f'tbpChoiceDecoding_fs100_ws0.1_we0.2_trial_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.2_we0.3_trial_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.3_we0.4_trial_decision.h.m_DRops.dim{ndim}.ddr',
    f'tbpChoiceDecoding_fs100_ws0.4_we0.5_trial_decision.h.m_DRops.dim{ndim}.ddr',
]

# control analysis where we start from the beginning of the trial, regardless of where the target comes
# catch_modellist = [
#     f'tbpChoiceDecoding_fs100_ws0.0_we0.1_trial_fromfirst_decision.cr.ich_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.1_we0.2_trial_fromfirst_decision.cr.ich_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.2_we0.3_trial_fromfirst_decision.cr.ich_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.3_we0.4_trial_fromfirst_decision.cr.ich_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.4_we0.5_trial_fromfirst_decision.cr.ich_DRops.dim{ndim}.ddr',
# ]
# target_modellist = [
#     f'tbpChoiceDecoding_fs100_ws0.0_we0.1_trial_fromfirst_decision.h.m_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.1_we0.2_trial_fromfirst_decision.h.m_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.2_we0.3_trial_fromfirst_decision.h.m_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.3_we0.4_trial_fromfirst_decision.h.m_DRops.dim{ndim}.ddr',
#     f'tbpChoiceDecoding_fs100_ws0.4_we0.5_trial_fromfirst_decision.h.m_DRops.dim{ndim}.ddr',
# ]


if models == "targets":
    modellist = target_modellist
    bad_sites = bad_target_sites + bad_sites
elif models == "catch":
    modellist = catch_modellist
    bad_sites = bad_catch_sites + bad_sites
else:
    raise ValueError("Need to specific models as either 'catch' or 'targets'")

if shuffle:
    modellist = [m+"_shuffle" for m in modellist]

if regress_pupil:
    modellist = [m+"_PR" for m in modellist]

if FA:
    # null model
    m0 = [m+"_FAperstim.0" for m in modellist]
    # no correlation models
    m1 = [m+"_FAperstim.5" for m in modellist]
    m2 = [m+"_FAperstim.6" for m in modellist]
    
    # "normal" models
    m3 = [m+"_FAperstim.1" for m in modellist]
    m4 = [m+"_FAperstim.2" for m in modellist]
    m5 = [m+"_FAperstim.3" for m in modellist]
    m6 = [m+"_FAperstim.4" for m in modellist]
    modellist = m0 + m1 + m2 + m3 + m4 + m5 + m6

sites = [s for s in sites if s not in bad_sites]

nd.enqueue_models(celllist=sites,
                        batch=batch,
                        modellist=modellist,
                        executable_path=python_path,
                        script_path=script,
                        user='hellerc',
                        force_rerun=force_rerun,
                        reserve_gb=2)