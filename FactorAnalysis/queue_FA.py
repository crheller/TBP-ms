import nems0.db as nd
import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms/")
from settings import BAD_SITES

batches = [324]
modelnames = ["FA_perstim"]
force_rerun = True

for batch in batches:

    sites, _ = nd.get_batch_sites(batch)
    sites = [s for s in sites if s not in BAD_SITES]

    script = '/auto/users/hellerc/code/projects/TBP-ms/FactorAnalysis/cache_FA_results.py'
    python_path = '/auto/users/hellerc/miniconda3/envs/lbhb/bin/python'
    nd.enqueue_models(celllist=sites,
                    batch=batch,
                    modellist=modelnames,
                    executable_path=python_path,
                    script_path=script,
                    user='hellerc',
                    force_rerun=force_rerun,
                    reserve_gb=2, priority=2)