"""
Cache behavioral performance for each animal over training
Save dprime for each target / animal / session
Save vector of RTs for each target / animal (concatenated across sessions)
"""
import json
import nems_lbhb.tin_helpers as thelp
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.behavior import get_reaction_times
import nems0.db as nd
import os
import numpy as np
import datetime as dt

# TODO - fix Clathrus / figure out why / how many are failing


# globals
results_path = "/auto/users/hellerc/results/TBP-ms/behavior"
min_trials = 50
runclass = "TBP"
snr_strs = ['-inf', '-10', '-5', '0', '5', 'inf'] # store results sorted by SNR
options = {
            "resp": False, 
            "pupil": False, 
            "rasterfs": 20, 
            "keep_following_incorrect_trial": True, 
            "keep_cue_trials": True, 
            "keep_early_trials": True
}

# name, start date, end date
animals = [
    ("Armillaria", "2020-10-05", "2020-10-30"),
    ("Cordyceps", "2020-08-05", "2020-09-24"),
    ("Jellybaby", "2021-03-10", "2021-05-29"),
    ("Clathrus", "2021-12-08", "2022-06-01")
]
animals = [
    ("Clathrus", "2022-02-23", "2022-06-01")
]

for an in animals:
    print(f"Analyzing animal {an[0]}\n\n")
    animal = an[0]
    ed = an[1]
    ld = an[2]

    ed = dt.datetime.strptime(ed, '%Y-%m-%d')
    ld = dt.datetime.strptime(ld, '%Y-%m-%d')

    # get list of parmfiles and sort by date (all files from one day go into one analysis??)
    sql = f"SELECT gDataRaw.resppath, gDataRaw.parmfile, pendate FROM gPenetration INNER JOIN gCellMaster ON (gCellMaster.penid = gPenetration.id)"\
                    f" INNER JOIN gDataRaw ON (gCellMaster.id = gDataRaw.masterid) WHERE" \
                    f" gDataRaw.runclass='{runclass}' and gDataRaw.bad=0 and gDataRaw.trials>{min_trials} and"\
                    f" gPenetration.animal = '{animal}' and gDataRaw.behavior='active'"
    d = nd.pd_query(sql)
    d['date'] = d['pendate'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))

    # screen for date
    d = d[(d['date'] >= ed) & (d['date'] <= ld)]
    d = d.sort_values(by='date')

    # join path
    d['parmfile_path'] = [os.path.join(d['resppath'].iloc[i], d['parmfile'].iloc[i]) for i in range(d.shape[0])]

    # define set of unique dates
    uDate = d['date'].unique()

    rts = {k: [] for k in snr_strs}
    dprime = {k: [] for k in snr_strs if ('-inf' not in k)}
    DI = {k: [] for k in snr_strs if ('-inf' not in k)}
    nSessions = {k: 0 for k in snr_strs}
    bad_day = []
    for ud in uDate:
        print(f"Analyzing date: {ud}\n")
        parmfiles = d[d.date==ud].parmfile_path.values.tolist()
        # add catch to make sure "siteid" the same for all files
        sid = [p.split(os.path.sep)[-1][:7] for p in parmfiles]
        if np.any(np.array(sid) != sid[0]):
            bad_idx = (np.array(sid)!=sid[0])
            parmfiles = np.array(parmfiles)[~bad_idx].tolist()
        
        try:
            manager = BAPHYExperiment(parmfiles)
        except IndexError:
            print(f"\n can't load {parmfiles}. Not flushed?")
            continue 

        # make sure only loaded actives
        pf_mask = [True if k['BehaveObjectClass']=='RewardTargetLBHB' else False for k in manager.get_baphy_exptparams()]
        if sum(pf_mask) == len(manager.parmfile):
            pass
        else:
            parmfiles = np.array(manager.parmfile)[pf_mask].tolist()
            manager = BAPHYExperiment(parmfiles)

        try:
            # get behavior performance
            performance = manager.get_behavior_performance(**options)

            # get reaction times of targets, only for "valid" trials
            bev = manager.get_behavior_events(**options)
            bev = manager._stack_events(bev)
            bev = bev[bev.invalidTrial==False]
            _rts = get_reaction_times(manager.get_baphy_exptparams()[0], bev, **options)

            targets = _rts['Target'].keys()
            cat = [t for t in targets if '-Inf' in t][0]
            snrs = thelp.get_snrs(targets)
            # keep only the freqs with same CF as catch
            freqs = thelp.get_freqs(targets)
            idx = [True if freq==thelp.get_freqs([cat])[0] else False for freq in freqs]
            targets = np.array(list(targets))[idx].tolist()
            snrs = [s for s, i in zip(snrs, idx) if i==True]
            untar = []
            for s, t in zip(snrs, targets):
                rts[str(s)].extend(_rts['Target'][t])
                _t = t.split(':')[0]
                if ('-Inf' not in _t) & (_t not in untar):
                    nSessions[str(s)] += 1
                    untar.append(_t)
                    try:
                        dprime[str(s)].extend([performance['dprime'][_t+'_'+cat.split(':')[0]]])
                        DI[str(s)].extend([performance['LI'][_t+'_'+cat.split(':')[0]]])
                    except:
                        # probably a reminder target
                        dprime[str(s)].extend([performance['dprime'][_t+'+reminder_'+cat.split(':')[0]]])
                        DI[str(s)].extend([performance['LI'][_t+'+reminder_'+cat.split(':')[0]]])
        except IndexError:
            # probably mismatch between pump dur and number of targets
            print(f"{ud} is a 'bad' day \n")
            bad_day.append(ud)

    # save results for this animal
    results = {
        "RTs": rts,
        "dprime": dprime,
        "DI": DI,
        "n": nSessions   
    }
    json.dump( results, open( os.path.join(results_path, f"{animal}_training.json"), 'w' ) )