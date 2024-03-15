"""
For "trial" decoding
t = 0.0 is aligned to (target/catch) sound onset
so ws0.0_we0.1_trial should be equivalent to ws0.0_we0.1 etc.

check that here to make sure it worked as expected.
"""
import matplotlib.pyplot as plt
import pandas as pd
import nems_lbhb.tin_helpers as thelp
import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from path_helpers import results_file
from settings import RESULTS_DIR, BAD_SITES
import numpy as np
import nems0.db as nd
import scipy.stats as ss
import scipy.ndimage as sd

normal_model = "tbpChoiceDecoding_fs100_ws0.0_we0.1_decision.cr.ich_DRops.dim2.ddr"
trial_model = "tbpChoiceDecoding_fs100_ws0.0_we0.1_trial_decision.cr.ich_DRops.dim2.ddr"

batch = 324
sites, _ = nd.get_batch_sites(batch)

nn = []
tt = []
nnd = []
ttd = []
for site in sites:
    try:
        nres = pd.read_pickle(results_file(RESULTS_DIR, site, batch, normal_model, "output.pickle"))   
        tres = pd.read_pickle(results_file(RESULTS_DIR, site, batch, trial_model, "output.pickle"))   

        nn.extend(nres["percent_correct"].values.tolist())
        tt.extend(tres["percent_correct"].values.tolist())
        nnd.extend(nres["dp"].values.tolist())
        ttd.extend(tres["dp"].values.tolist())
    except:
        print(f"no model for site: {site}")


f, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].scatter(nn, tt, s=10)
ax[0].set_xlabel("normal")
ax[0].set_ylabel("trial based")
ax[0].set_xlim((0.5, 1))
ax[0].set_ylim((0.5, 1))
ax[0].set_title("perc. correct")

ul = np.max(nnd+ttd)
ax[1].scatter(nnd, ttd, s=10)
ax[1].set_xlabel("normal")
ax[1].set_ylabel("trial based")
ax[1].set_title("d-prime")
ax[1].set_aspect("equal")
ax[1].set_ylim((0, ul)); ax[1].set_xlim((0, ul))

f.tight_layout()