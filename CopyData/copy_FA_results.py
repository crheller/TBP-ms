"""
Loop over FA results and copy them to the dryad location
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
    "FA_perstim",
    "FA_perstim_PR"
]

sites = [s for s in sites if s not in BAD_SITES]
for site in sites:
    for model in modellist:
        f = os.path.join(LBHB_DIR, "factor_analysis", str(batch), site, model+".pickle")
        if os.path.isfile(f):
            newfile = f.replace(LBHB_DIR, RESULTS_DIR)
            newfile = newfile.replace(f"factor_analysis/{batch}/", "")
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