# plot penetration maps for each recording site
# except for CRD - didn't save stereotactic coords for him
import os
rdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
import sys
sys.path.append(rdir)
from settings import RESULTS_DIR
from plotting import penetration_map
import pandas as pd
import numpy as np

# =========================================================================
# Jellybaby
df = pd.read_csv(os.path.join(RESULTS_DIR, "JLY_LH_penMap.csv"), index_col=0)

coords = [np.array([df.iloc[i]["coord1"], df.iloc[i]["coord2"], df.iloc[i]["coord3"]]) for i in range(df.shape[0])]
area = df["area"]
area[area.isna()] = "NA"
area = area.tolist()
bf = df["BF"].tolist()
sites = df["site"].tolist() 

fig, coords = penetration_map(sites, area, bf, coords, equal_aspect=True, flip_X=True, flatten=True)
fig.axes[0].grid()