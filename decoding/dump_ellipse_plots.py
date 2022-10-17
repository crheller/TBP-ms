"""
For each batch 324 site, generate active vs. passive ellipse plots
showing target / catch responses in the dDR space
"""

import charlieTools.TBP_ms.loaders as loaders
import charlieTools.TBP_ms.plotting as plotting
import nems0.db as nd
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb import tin_helpers as thelp

import sys
sys.path.append("/auto/users/hellerc/code/projects/TBP-ms")
from settings import BAD_SITES
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


figpath = "/auto/users/hellerc/code/projects/TBP-ms/temp_figs/ellipse/"

batch = 324
mask = ["HIT_TRIAL", "CORRECT_REJECT_TRIAL", "PASSIVE_EXPERIMENT", "MISS_TRIAL"]
sites = nd.get_batch_sites(324)[0]
sites = [s for s in sites if s not in BAD_SITES]
for site in sites:
    filename = os.path.join(figpath, f"{site}_ellipse.png")
    plotting.dump_ellipse_plot(site, batch, filename=filename, mask=mask)