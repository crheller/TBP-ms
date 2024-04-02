import nems0.db as nd
import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms/")
from settings import BAD_SITES

batches = [324]
modelnames = ["FA_perstim_PR"]
modelnames = [
    "FA_perstim_choice",
    "FA_perstim_choice_ws0.0_we0.1_trial",
    "FA_perstim_choice_ws0.1_we0.2_trial",
    "FA_perstim_choice_ws0.2_we0.3_trial",
    "FA_perstim_choice_ws0.3_we0.4_trial",
    "FA_perstim_choice_ws0.4_we0.5_trial",
    "FA_perstim_choice_ws0.0_we0.1_trial_fromfirst",
    "FA_perstim_choice_ws0.1_we0.2_trial_fromfirst",
    "FA_perstim_choice_ws0.2_we0.3_trial_fromfirst",
    "FA_perstim_choice_ws0.3_we0.4_trial_fromfirst",
    "FA_perstim_choice_ws0.4_we0.5_trial_fromfirst",
]
# analysis_script = '/auto/users/hellerc/code/projects/TBP-ms/FactorAnalysis/cache_FA_results.py'
analysis_script = '/auto/users/hellerc/code/projects/TBP-ms/FactorAnalysis/cache_FA_choice.py'
force_rerun = True

bad_target_sites = [
    "CLT009a",
    "CRD019b",
    "JLY008b",
    "JLY010b",
    "JLY014d"
]

for batch in batches:

    sites, _ = nd.get_batch_sites(batch)
    sites = [s for s in sites if s not in BAD_SITES+bad_target_sites]
    sites = ["CLT001a"]

    script = analysis_script
    python_path = '/auto/users/hellerc/miniconda3/envs/lbhb/bin/python'
    nd.enqueue_models(celllist=sites,
                    batch=batch,
                    modellist=modelnames,
                    executable_path=python_path,
                    script_path=analysis_script,
                    user='hellerc',
                    force_rerun=force_rerun,
                    reserve_gb=2, priority=2)