"""
For each recording session (with neural data)
save behavioral performance for each target.

Use these to relate single target neural d' values to 
single target behavioral d' values
"""
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems0.db as nd
import os
import pandas as pd
import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from settings import BAD_SITES, RESULTS_DIR

batch = 324
sites = nd.get_batch_sites(batch)[0]
sites = [s for s in sites if s not in BAD_SITES]

options1 = {
            "resp": False, 
            "pupil": False, 
            "rasterfs": 20, 
            "keep_following_incorrect_trial": True, 
            "keep_cue_trials": True, 
            "keep_early_trials": True
}
options2 = {
            "resp": False, 
            "pupil": False, 
            "rasterfs": 20, 
            "keep_following_incorrect_trial": False, 
            "keep_cue_trials": False, 
            "keep_early_trials": False
}

d1 = pd.DataFrame(columns=["dprime", "e1", "e2", "site"])
d2 = pd.DataFrame(columns=["dprime", "e1", "e2", "site"])
ii = 0
jj = 0
for site in sites:
    print(f"\n \n {site} \n \n")
    manager = BAPHYExperiment(cellid=site, batch=batch)
    perf1 = manager.get_behavior_performance(**options1)
    kk = [k for k in perf1["RR"].keys() if (k != "Reference") & ("reminder" not in k)]
    cc = [c for c in kk if "-Inf" in c]
    kk = [k for k in kk if k not in cc]
    for k in kk:
        for c in cc:
            s = k+"_"+c
            d1.loc[ii, :] = [perf1["dprime"][s], k, c, site]
            ii += 1

    perf2 = manager.get_behavior_performance(**options2)
    kk = [k for k in perf2["RR"].keys() if (k != "Reference") & ("reminder" not in k)]
    cc = [c for c in kk if "-Inf" in c]
    kk = [k for k in kk if k not in cc]
    for k in kk:
        for c in cc:
            s = k+"_"+c
            d2.loc[ii, :] = [perf2["dprime"][s], k, c, site]
            jj += 1

respath = os.path.join(RESULTS_DIR, "behavior_recordings")
d1.to_pickle(os.path.join(respath, "all_trials.pickle"))
d2.to_pickle(os.path.join(respath, "good_trials.pickle"))