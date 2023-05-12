import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import pickle
from tqdm import tqdm
from scipy.stats import lognorm

from standalone import run_network, extract_counts
from utils import connection_matrix


def extract_mean_current(current, time, nstim=30, stim_dur=1.1, discard=0.1):
    mean_currents = []
    
    for s in range(nstim):
        mask = (time > discard+s*stim_dur) & (time <= (s+stim_dur-discard)*stim_dur)
        mean_currents.append(current[mask].mean())
    
    return np.array(mean_currents)

def get_currents(xc, rec_rel, max_input, min_input=0.4, n_stims=30):
    stims = np.linspace(min_input, max_input, n_stims)

    all_trials = []

    seed = 0

    np.random.seed(seed=0)
    cm = connection_matrix(xc=xc/100, eps=0.2)

    stim_dur = 2.1

    tdi = TimedArray(100*stims/xc * Hz, dt=stim_dur*second)

    exc_spikes, inh_spikes, Na_current, _ = run_network(tdi=tdi, conn=cm,
                                                        g=20, mode='nonadapt',
                                                        state=True, inh_threshold_diff=5,
                                                        rec_rel=rec_rel,
                                                        tot_time=n_stims*stim_dur, seed_num=seed)
    currents = extract_mean_current(*Na_current, stim_dur=stim_dur, nstim=n_stims)
    return currents


if __name__ == '__main__':
    current_lists_all = {}

    n_stims = 30

    rec_rel_list = [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2]
    max_intensity_lists = {
        1:  [0.9, 1.0, 1.2, 1.5, 1.8, 2.4, 3.5, 4.5, 6.5, 10, 18.8, 34],
        2:  [0.9, 1.0, 1.2, 1.5, 1.8, 2.4, 3.5, 4.5, 6.5, 10, 18.8, 34],
        3:  [0.9, 1.0, 1.2, 1.5, 1.8, 2.4, 3.5, 4.5, 6.5, 10, 18.8, 34],
        5:  [0.9, 1.0, 1.2, 1.5, 1.8, 2.4, 3.5, 4.5, 6.5, 10, 18.8, 34],
        10:  [0.9, 1.0, 1.23, 1.55, 1.9, 2.55, 3.7, 4.95, 7.1, 11.1, 20.6, 36],
        15:  [0.9, 1.0, 1.23, 1.55, 1.97, 2.6, 3.88, 5.25, 7.6, 12, 21.4, 36.4],
        20: [0.9, 1.0, 1.25, 1.6, 2, 2.75, 4.1, 5.6, 8.1, 13, 21, 38],
        50: [0.9, 1.0, 1.3, 1.7, 2.2, 3.1, 4.7, 6.2, 9.1, 14, 23, 38],
        100: [0.9, 1.02, 1.4, 1.86, 2.5, 3.5, 5.3, 7.1, 10.7, 16.4, 25, 42.2]
    }


    # for xc in [5, 10, 15, 20, 50, 100]:
    for xc in [2,3]:
        currents_list = {}
        current_lists_all[xc] = currents_list
        
        for rec_rel, max_input in tqdm(zip(rec_rel_list, max_intensity_lists[xc]), total=len(rec_rel_list)):
            
            currents = get_currents(xc, rec_rel, max_input)
            currents_list[rec_rel] = currents
            
        pickle.dump(currents_list, open( f"data standalone/currents/Na_currents_xc{xc}.p", "wb" ) )