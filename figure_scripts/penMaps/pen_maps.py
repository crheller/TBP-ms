# plot penetration maps for each recording site
# except for CRD - didn't save stereotactic coords for him

from nems_lbhb.penetration_map import penetration_map

import matplotlib.pyplot as plt


# =========================================================================
# Jellybaby
# RH
ref = [0.91, 5.27, 4.99]
tr = [42,0]
sites = ['JLY002', 
        'JLY003',
        'JLY004',
        'JLY007d',
        'JLY008b',
        'JLY009b',
        'JLY010b',
        'JLY011c',
        'JLY012d',
        'JLY013c', 
        'JLY014d', 
        'JLY015a',
        'JLY016a',
        'JLY017a',
        'JLY018a',
        'JLY019a']
landmarks = {'MidLine'     : ref+[1.384, 4.53, 4.64]+tr,
              'OccCrest': ref+[0.076, 5.27, 5.28]+tr,
              'Occ_Crest_in' : ref+[0.490, 5.27, 5.28]+tr}
fig, coords = penetration_map(sites, equal_aspect=True, flip_X=False, flatten=True, landmarks=None)#landmarks)
fig.axes[0].grid()

# LH
sites = [
    'JLY020', 
    'JLY021c', 
    'JLY022b',
    'JLY023d',
    'JLY024c',
    'JLY025c',
    'JLY026f',
    'JLY028a',
    'JLY029e',
    'JLY030c',
    'JLY031c',
    'JLY032b',
    'JLY033b',
    'JLY034a',
    'JLY035d',
    'JLY036a',
    'JLY037a',
    'JLY038a',
    'JLY039b',
    'JLY040b',
    'JLY041a',
    'JLY042a',
    'JLY043a',
    'JLY044a',
    'JLY045a',
    'JLY046a'
]
landmarks = None
fig, coords = penetration_map(sites, equal_aspect=True, flip_X=True, flatten=True, landmarks=landmarks)
fig.axes[0].grid()
#fig.axes[0].invert_yaxis()

# =========================================================================
# Armillaria
# LH
sites = [
    "ARM033a",
    "ARM032a",
    "ARM031a",
    "ARM030a",
    "ARM029a",
    "ARM028a",
    "ARM027a",
    "ARM026a",
    "ARM025a",
    "ARM024a",
    "ARM023a",
    "ARM022a",
    "ARM021a",
    "ARM020a",
    "ARM019a",
    "ARM018a",
    "ARM017a",
    "ARM016c",
    "ARM015b",
    "ARM014b",
    "ARM013b",
    "ARM012d",
]

fig, coords = penetration_map(sites, equal_aspect=True, flip_X=True, flatten=True, landmarks=None)
fig.axes[0].grid()

# RH -- Didn't record positions, can't plot

# =========================================================================
# Clathrus -- BFs not saved yet.
# LH
sites = [
    "CLT053a",
    "CLT052a",
    "CLT051a",
    "CLT050a",
    "CLT049a"
]
