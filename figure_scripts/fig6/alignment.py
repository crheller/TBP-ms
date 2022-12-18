"""
Compare first FA loading to the delta mu axis
for each target (vs. catch)
"""
import nems0.db as nd

import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
import os
import sys
from path_helpers import results_file
from settings import RESULTS_DIR, BAD_SITES

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

batch = 324
sqrt = True
sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])
amodel = 'tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'
pmodel = 'tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise'

sites = [s for s in sites if s not in BAD_SITES]
rra = 0
rrp = 0
dfa = pd.DataFrame(columns=["acos_sim", "e1", "e2", "area", "site"])
dfp = pd.DataFrame(columns=["pcos_sim", "e1", "e2", "area", "site"])
for site in sites:
    d = pd.read_pickle(os.path.join(RESULTS_DIR, "factor_analysis", str(batch), site, "FA_perstim.pickle"))
    area = nd.pd_query(sql="SELECT area from sCellFile where cellid like %s", params=(f"%{site}%",)).iloc[0][0]
    
    # load decoding results
    ares = pd.read_pickle(results_file(RESULTS_DIR, site, batch, amodel, "output.pickle"))
    pres = pd.read_pickle(results_file(RESULTS_DIR, site, batch, pmodel, "output.pickle"))
    for e in [k for k in d["active"].keys() if 'CAT' in k]:
        afa = d["active"][e]["components_"][0, :]
        pfa = d["passive"][e]["components_"][0, :]
        try:
            mm = (ares["e1"]==e) & (ares["e2"].str.startswith("TAR"))
            for i in range(sum(mm)):
                _du = ares[mm]["dU"].iloc[i]
                dua = (_du / np.linalg.norm(_du)).dot(ares[mm]["dr_loadings"].iloc[0]).squeeze()
                dfa.loc[rra, :]  = [abs(afa.dot(dua)), e, ares[mm]["e2"].iloc[i], area, site]
                rra += 1

            mm = (pres["e1"]==e) & (pres["e2"].str.startswith("TAR"))
            for i in range(sum(mm)):
                _dup = pres[mm]["dU"].iloc[i]
                dup = (_dup / np.linalg.norm(_dup)).dot(pres[mm]["dr_loadings"].iloc[0]).squeeze()
                dfp.loc[rrp, :]  = [abs(pfa.dot(dup)), e, pres[mm]["e2"].iloc[i], area, site]
                rrp += 1

        except IndexError:
            print(f"didn't find matching decoding entry for {e}, {site}")
    
# merge 
df = dfa.merge(dfp, on=["e1", "e2", "area", "site"])


f, ax = plt.subplots(1, 2, figsize=(4, 4), sharey=True)

for i, a in enumerate(["A1", "PEG"]):
    y = df[df.area==a]["acos_sim"]
    ax[i].errorbar(0, y.mean(), yerr=y.std()/np.sqrt(len(y)), marker="o",
            capsize=2, markeredgecolor="k", label="active") 
    y = y = df[df.area==a]["pcos_sim"]
    ax[i].errorbar(1, y.mean(), yerr=y.std()/np.sqrt(len(y)), marker="o",
            capsize=2, markeredgecolor="k", label="passive") 
    ax[i].set_title(a)
    ax[i].set_xticks([])
ax[0].set_ylabel("Cos. similarity (dU vs. FA1)")
ax[0].legend(frameon=False)
f.tight_layout()