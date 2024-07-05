"""
Loop over decoding results and copy them to the dryad location
"""
import nems0.db as nd
import sys
import os
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from helpers.path_helpers import results_file
from settings import RESULTS_DIR, BAD_SITES
import numpy as np

batch = 324
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])
LBHB_DIR = "/auto/users/hellerc/results/TBP-ms"

modellist = [
    # Standard active / passive decoding jobs
    'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise', 
    'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise',
    # Standard active / passive decoding jobs with pupil regressed out
    'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR',
    'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR',
    # Factor analysis simulations
    'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR_FAperstim.0.PR',
    'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR_FAperstim.1.PR',
    'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR_FAperstim.3.PR',
    'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR_FAperstim.4.PR',
    'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR_FAperstim.0.PR',
    'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR_FAperstim.1.PR',
    'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR_FAperstim.3.PR',
    'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise_PR_FAperstim.4.PR',

    # choice decoding models
    # at beginning of trial
    'tbpChoiceDecoding_fs10_ws0.0_we0.1_trial_fromfirst_decision.h.m_DRops.dim2.ddr',
    'tbpChoiceDecoding_fs10_ws0.1_we0.2_trial_fromfirst_decision.h.m_DRops.dim2.ddr',
    'tbpChoiceDecoding_fs10_ws0.2_we0.3_trial_fromfirst_decision.h.m_DRops.dim2.ddr',
    'tbpChoiceDecoding_fs10_ws0.3_we0.4_trial_fromfirst_decision.h.m_DRops.dim2.ddr',
    'tbpChoiceDecoding_fs10_ws0.4_we0.5_trial_fromfirst_decision.h.m_DRops.dim2.ddr',
    # during target / catch (end of trial)
    'tbpChoiceDecoding_fs10_ws0.0_we0.1_trial_decision.h.m_DRops.dim2.ddr',
    'tbpChoiceDecoding_fs10_ws0.1_we0.2_trial_decision.h.m_DRops.dim2.ddr',
    'tbpChoiceDecoding_fs10_ws0.2_we0.3_trial_decision.h.m_DRops.dim2.ddr',
    'tbpChoiceDecoding_fs10_ws0.3_we0.4_trial_decision.h.m_DRops.dim2.ddr',
    'tbpChoiceDecoding_fs10_ws0.4_we0.5_trial_decision.h.m_DRops.dim2.ddr',
]

sites = [s for s in sites if s not in BAD_SITES]
active = []
passive = []
for site in sites:
    for model in modellist:
        f = results_file(LBHB_DIR, site, batch, model, "output.pickle")
        if os.path.isfile(f):
            newfile = f.replace(LBHB_DIR, RESULTS_DIR)
            newfile = newfile.replace(f"{batch}/", "")
            if os.path.isdir(os.path.dirname(newfile)):
                os.system(f"cp {f} {newfile}")
            elif os.path.isdir(os.path.dirname(os.path.dirname(newfile))):
                os.system(f"mkdir {os.path.join(RESULTS_DIR, site, model)}")
                os.system(f"cp {f} {newfile}")
            elif os.path.isdir(os.path.dirname(os.path.dirname(os.path.dirname(newfile)))):
                os.system(f"mkdir {os.path.join(RESULTS_DIR, site)}")
                os.system(f"mkdir {os.path.join(RESULTS_DIR, site, model)}")
                os.system(f"cp {f} {newfile}")
        else:
            print(f"{model} not found for site: {site}\n")