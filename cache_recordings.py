"""
Cache recordings to be uploaded to dryad
Also build a mini "database" of the datasets to use for querying datasets
"""
import nems0.db as nd
from nems_lbhb.baphy_experiment import BAPHYExperiment
from settings import BAD_SITES, RESULTS_DIR
import pandas as pd
import os

recordings_dir = os.path.join(RESULTS_DIR, "recordings")

sites = nd.get_batch_sites(324)[0]
sites = [s for s in sites if s not in BAD_SITES]

uris_50hz = []
uris_10hz =[]
df = pd.DataFrame(columns=["site", "area", "10hz_uri", "50hz_uri"], index=sites)
for site in sites:
    print(f"{site}\n")
    area = nd.pd_query(sql=f"SELECT area from sCellFile where cellid like '%{site}%'").iloc[0][0]
    batch = 324
    
    # 50 Hz sampling
    fs = 50
    options = {'resp': True, 'pupil': True, 'rasterfs': fs, 'stim': False}
    manager = BAPHYExperiment(batch=batch, cellid=site, rawid=None)
    uri = manager.get_recording_uri(recache=False, **options)
    uris_50hz.append(uri)
    hash_50hz = uri.split(os.path.sep)[-1]

    # 10Hz sampling
    fs = 10
    options = {'resp': True, 'pupil': True, 'rasterfs': fs, 'stim': False}
    manager = BAPHYExperiment(batch=batch, cellid=site, rawid=None)
    uri = manager.get_recording_uri(recache=False, **options)
    uris_10hz.append(uri)
    hash_10hz = uri.split(os.path.sep)[-1]

    df.loc[site, :] = [site, area, hash_10hz, hash_50hz]


# for uri in list, copy them to the dryad folder
for i, (uri10, uri50) in enumerate(zip(uris_10hz, uris_50hz)):
    print(f"saving {i}/{len(uris_10hz)}")
    dest = uri10.replace("/auto/data/nems_db/recordings/324", recordings_dir)
    os.system(f"cp {uri10} {dest}")

    dest = uri50.replace("/auto/data/nems_db/recordings/324", recordings_dir)
    os.system(f"cp {uri50} {dest}")