"""
This script runs the simulation and saves the average synaptic currents.
"""

from brian2 import *
import numpy as np
from tqdm import tqdm
import os
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.ndimage import gaussian_filter1d
import pickle
from utils import connection_matrix, extract_mean_current
import warnings
from time import time as wall_time
# set_device('cpp_standalone', build_on_run=False)
from network import extract_counts, extract_mean_current, run_network
    

if __name__ == '__main__':
    BrianLogger.suppress_hierarchy('brian2.codegen.generators.base',
                               filter_log_file=True)
    BrianLogger.suppress_hierarchy('brian2.devices.cpp_standalone.device.delete_skips_directory',
                                   filter_log_file=True)
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--nruns', type=int, default=36)
    parser.add_argument('--corr', type=float, default=0.05)
    parser.add_argument('--min', type=float, default=0.)
    parser.add_argument('--max', type=float, default=0.9)
    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--group', type=str)
    parser.add_argument('--strength', type=float, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inh', type=int, default=20)
    parser.add_argument('--stims', type=int, default=30)
    parser.add_argument('--bcgexc', type=float, default=0.5)
    parser.add_argument('--bcginh', type=float, default=0.125)
    parser.add_argument('--repeats', type=int, default=1)
    
    args = parser.parse_args()
    
    np.random.seed(seed=args.seed)
    group = args.group
    
    cm_seed = args.seed
    xc = args.corr
    np.random.seed(seed=cm_seed)
    cm = connection_matrix(xc=xc, eps=0.2)
    
    stims = np.linspace(args.min, args.max, args.stims)
    tdi_spikes = TimedArray(np.tile(stims/xc, args.repeats) * Hz, dt=1.1*second)

    all_trials_exc = []
    all_trials_inh = []

    
    simulation_kwargs = dict(
        conn=cm,
        g=args.inh,
        mode='nonadapt',
        inh_threshold_diff=5,
        rec_rel=args.strength,
        background_exc=args.bcgexc,
        background_inh=args.bcginh
    )
    
    stim_dur = 5.1
    tdi_cost = TimedArray(stims/xc * Hz, dt=stim_dur*second)

    exc_spikes, inh_spikes, Na_currents, currents = run_network(**simulation_kwargs, tdi=tdi_cost, state=True,
                                             tot_time=len(stims)*stim_dur, seed_num=42)
    
    exc_counts = extract_counts(*spikes_exc, 800, nstim=args.stims, n_repeat=args.repeats, stim_dur=stim_dur)
    inh_counts = extract_counts(*spikes_inh, 200, nstim=args.stims, n_repeat=args.repeats, stim_dur=stim_dur)
    
    # exc-exc, ext-exc, exc-inh, ext-inh, background-exc, background_inh
    keys = ['exc','ext','inh','ext_background','inh_background']
    
    currents_dict = {}
    curr_time = Na_currents[-1]
    
    for curr, name in zip(currents, keys):
        currents_dict[name] = extract_mean_current(curr, curr_time, stim_dur=stim_dur, nstim=len(stims))
    
    pickle.dump((stims, exc_counts, inh_counts, currents_dict),
                open( f"data/{group}_xc_{xc*100:.0f}_seed{cm_seed}.p", "wb" ) )
    
    os.system('rm -rf standalone\ codes/*')
    # print(all_trials.shape)
    # print(f"Done in {wall_time() - start_time:10.3f}")