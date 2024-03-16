"""
Manually recache recordings when somethign changed with the loader
"""

from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems0.db as nd
import numpy as np

batch = 324
fs = 10
recache = True

sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])


for site in [str(s) for s in sites]:
    options = {'resp': True, 'pupil': True, 'rasterfs': fs, 'stim': False}
    manager = BAPHYExperiment(batch=batch, cellid=site, rawid=None)
    rec = manager.get_recording(recache=recache, **options)
    rec['resp'] = rec['resp'].rasterize()