"""
Is there a difference in choice decoding between A1 and PEG?
Answer seems to be no, both have decent choice decoding at the population level.
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

catch_model = "tbpChoiceDecoding_decision.cr.ich_DRops.dim2.ddr"
target_model = "tbpChoiceDecoding_decision.h.m_DRops.dim2.ddr"

batch = 324
sqrt = True
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])
sites = [s for s in sites if s not in BAD_SITES]

catch_results = []
for site in sites:
    try:
        res = pd.read_pickle(results_file(RESULTS_DIR, site, batch, catch_model, "output.pickle"))    
        res["site"] = site
        area = nd.pd_query(sql="SELECT area from sCellFile where cellid like %s", params=(f"%{site}%",))
        area = area.iloc[0][0]
        res["area"] = area
        if sqrt:
            res["dp"] = np.sqrt(res["dp"])
        catch_results.append(res)
    except:
        print(f"model didn't exsit for site {site}. Prob too few reps")
catch_results = pd.concat(catch_results)

target_results = []
for site in sites:
    try:
        res = pd.read_pickle(results_file(RESULTS_DIR, site, batch, target_model, "output.pickle"))    
        res["site"] = site
        area = nd.pd_query(sql="SELECT area from sCellFile where cellid like %s", params=(f"%{site}%",))
        area = area.iloc[0][0]
        res["area"] = area
        if sqrt:
            res["dp"] = np.sqrt(res["dp"])
        target_results.append(res)
    except:
        print(f"model didn't exsit for site {site}. Prob too few reps")
target_results = pd.concat(target_results)


f, ax = plt.subplots(1, 2, figsize=(6, 6), sharey=True)

a1 = catch_results[catch_results.area=="A1"]["dp"]
x_a1 = np.random.normal(0, 0.1, len(a1))
ax[0].scatter(x_a1, a1, s=10, color="k")

peg = catch_results[catch_results.area=="PEG"]["dp"]
x_peg = np.random.normal(0, 0.1, len(peg)) + 1
ax[0].scatter(x_peg, peg, s=10, color="k")

ax[0].set_ylabel("d-prime (choice)")
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(["A1", "PEG"])
ax[0].set_title("Catch stimuli")

a1 = target_results[target_results.area=="A1"]["dp"]
x_a1 = np.random.normal(0, 0.1, len(a1))
ax[1].scatter(x_a1, a1, s=10, color="k")

peg = target_results[target_results.area=="PEG"]["dp"]
x_peg = np.random.normal(0, 0.1, len(peg)) + 1
ax[1].scatter(x_peg, peg, s=10, color="k")

ax[1].set_ylabel("d-prime (choice)")
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(["A1", "PEG"])
ax[1].set_title("Target stimuli")

f.tight_layout()