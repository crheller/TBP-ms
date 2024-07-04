"""
Cache final decoding models for all recording sites.

Original analysis ran parallelized on David lab compute cluster.
Reproduced here running all analyses in parallel. Could take a little while.
"""


modellist = [
    "tbpDecoding_mask.h.cr.m_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise", 
    "tbpDecoding_mask.pa_drmask.h.cr.m.pa_DRops.dim2.ddr-targetNoise",
]