"""
Runs the network with given parameters.
Saves the spike counts at different trials and the average Na current magnituted
for different stimuli
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

def extract_counts(spiketrain, indices, N, nstim=30, stim_dur=1.1, discard=0.1, n_repeat=1):
    counts_repeats = []
    
    for ii in range(n_repeat):
        counts_all_int = []

        for s in range(nstim):
            mask = (spiketrain > discard+s*stim_dur+ii*nstim*stim_dur) & (spiketrain <= (s+1)*stim_dur+ii*nstim*stim_dur)
            ixs, counts = np.unique(indices[mask], return_counts=True)
            counts_all = np.zeros(N)
            nonzero_mask = np.isin(np.arange(N), ixs)
            counts_all[nonzero_mask] = counts

            counts_all_int.append(counts_all)
        
        counts_repeats.append(np.array(counts_all_int))
    
    if n_repeat == 1:
        return np.array(counts_all_int)
    else:
        return counts_repeats

def run_network(eps=0.05, N_exc=800, N_inh=200, N_x=1000,
                J=0.001, g=5, rec_rel=1, inh_threshold_diff=0, conn=None,
                report=False, state=False, seed_num=42, mode='nonadapt',
                orientation=True, tot_time=12, tau_inh=5, p=1, tdi=None, g_exc=1,
                background_exc=0, background_inh=0):
    """
    if state is True, then the following is returned:
    1. excitatory spikes
    2. inhibitory spikes
    3. tuple with Na currents: exc-exc, ext-exc, exc-inh, ext-inh, background-exc, background_inh, time array
    4. actual currents: exc, ext, inh, ext_background, inh_background
    """
    
    pid = os.getpid()
    directory = f"standalone codes/standalone{pid}"
    set_device('cpp_standalone', directory=directory)
    
    seed(seed_num)
    np.random.seed(seed_num)
    
    Jx = J*uS
    Je = rec_rel*g_exc*J*uS
    Ji = rec_rel*g*J*uS
    
    Jee = Je
    Jei = Ji * 5/tau_inh
    Jie = Je
    Jii = Ji * 5/tau_inh
    
    taue = 5*ms
    taui = tau_inh*ms
    Ee = 0*mV
    Ei = -80*mV
    
    C = 150*pF
    gL = 10*nS
    EL = -80*mV
    Vth = -50*mV
    refrac = 5*ms
    tauw = 500*ms
    
    background_exc = background_exc*kHz
    background_inh = background_inh*kHz

    eqs_str = f'''
    dv/dt = (-gL*(v-EL) + Isyn) / C : volt (unless refractory)
    Isyn = -(ge+gx+geb)*(v-Ee)-(gi+gib)*(v-Ei) : amp
    
    dge/dt = -ge / taue : siemens
    dgx/dt = -gx / taue : siemens
    dgi/dt = -gi / taui : siemens
    
    mu_eb = Jx*taue*background_exc : siemens
    sig_eb = sqrt(background_exc*taue / 2) * Jx : siemens
    dgeb/dt = -(geb-mu_eb) / taue + sig_eb*sqrt(2/taue)*xi_exc : siemens
    
    mu_ib = Jx*g*taue*background_inh : siemens
    sig_ib = sqrt(background_inh*taui / 2) * Jx*g : siemens
    dgib/dt = -(gib-mu_ib) / taui + sig_ib*sqrt(2/taui)*xi_inh : siemens
    
    dIw/dt = (-Iw + a*(v-EL)) / tauw : amp
    ds/dt = -s / (tauw) : 1
    
    v_r : volt
    factor : 1
    '''
    
    eqs_exc = Equations(eqs_str, a=0*nS, ka=1*mV)
    eqs_inh = Equations(eqs_str, a=0*nS, ka=1*mV)
    
    if mode == 'ahp':
        reset_exc = '''
            v = EL
            Iw += 20*pA
        '''
    elif mode == 'dynthr':
        reset_exc = '''
            v = EL
            s += 3*0.05 * (1-s)
            Iw += 0*pA
        '''
    elif mode == 'nonadapt':
        reset_exc = '''
            v = EL
            Iw += 0*pA
        '''

#     G_exc = NeuronGroup(N_exc, eqs_exc,
#                 threshold='v > Vth+20*2*mV', reset=reset_exc, method='euler', refractory=refrac)
#     G_inh = NeuronGroup(N_inh, eqs_inh,
#                 threshold='v > Vth+20*0.5*mV', reset='v = EL', method='euler', refractory=refrac)

    G_exc = NeuronGroup(N_exc, eqs_exc,
                threshold='v > -55*mV', reset='v = EL', method='euler', refractory=refrac)
    G_inh = NeuronGroup(N_inh, eqs_inh,
                threshold='v > (-55-inh_threshold_diff)*mV', reset='v = EL', method='euler', refractory=refrac)
    
    G_exc.v = (np.random.rand(N_exc)*5)*mV + EL
    G_inh.v = (np.random.rand(N_inh)*5)*mV + EL
    
    # G_exc.v = EL
    # G_inh.v = EL
    
#     ii = np.arange(N_exc)
    
#     if orientation is True:
#         G_exc.factor = np.sin(ii*0.5*(np.pi)/N_exc)**2
#     else:
#         G_exc.factor = 1
    
#     ii = np.arange(N_inh)
    
#     if orientation is True:
#         G_inh.factor = np.sin(ii*0.5*(np.pi)/N_inh)**2
#     else:
#         G_inh.factor = 1
    
    PG = PoissonGroup(1000, rates='tdi(t)')
        
    
    Sex = Synapses(PG, G_exc, on_pre='gx += Jx')
    Six = Synapses(PG, G_inh, on_pre='gx += Jx')
    Sie = Synapses(G_exc, G_inh, on_pre='ge += Jie*(rand()<p)')
    See = Synapses(G_exc, G_exc, on_pre='ge += Jee*(rand()<p)')
    Sei = Synapses(G_inh, G_exc, on_pre='gi += Jei*(rand()<p)')
    Sii = Synapses(G_inh, G_inh, on_pre='gi += Jii*(rand()<p)')
    
    if conn is None:
        Sex.connect(p=xc)
        Six.connect(p=xc)
        Sie.connect(p=eps)
        See.connect(p=eps)
        Sei.connect(p=eps)
        Sii.connect(p=eps)
    else:
        # ext to exc
        sources, targets = conn[:N_x,N_x:N_x+N_exc].nonzero()
        if len(sources) > 0:
            Sex.connect(i=sources, j=targets)
        else:
            Sex.connect(p=0)
            
        # ext to inh
        sources, targets = conn[:N_x,N_x+N_exc:].nonzero()
        if len(sources) > 0:
            Six.connect(i=sources, j=targets)
        else:
            Six.connect(p=0)
        
        # exc to exc
        sources, targets = conn[N_x:N_x+N_exc,N_x:N_x+N_exc].nonzero()
        if len(sources) > 0:
            See.connect(i=sources, j=targets)
#             See.connect(p=0)
        else:
            See.connect(p=0)
        
        # exc to inh
        sources, targets = conn[N_x:N_x+N_exc,N_x+N_exc:].nonzero()
        if len(sources) > 0:
            Sie.connect(i=sources, j=targets)
#             Sie.connect(p=0)
        else:
            Sie.connect(p=0)
        
        # inh to exc
        sources, targets = conn[N_x+N_exc:,N_x:N_x+N_exc].nonzero()
        if len(sources) > 0:
            Sei.connect(i=sources, j=targets)
#             Sei.connect(p=0)
        else:
            Sei.connect(p=0)
        
        # inh to inh
        sources, targets = conn[N_x+N_exc:,N_x+N_exc:].nonzero()
        if len(sources) > 0:
            Sii.connect(i=sources, j=targets)
#             Sii.connect(p=0)
        else:
            Sii.connect(p=0)
            
    
    spike_monitors = [SpikeMonitor(G_exc), SpikeMonitor(G_inh)]

    net = Network(collect())
    net.add(spike_monitors)
    
    if state:
        state_monitors = [StateMonitor(G_exc, ['v','ge','gi','gx','geb','gib'], record=True), StateMonitor(G_inh, ['v','ge','gi','gx','geb','gib'], record=True)]
        net.add(state_monitors)
    
    
    seed(seed_num)
    np.random.seed(seed_num)
    
    if report:
        net.run(tot_time*second, report='stderr')
    else:
        net.run(tot_time*second)
    
    spikes_exc = (spike_monitors[0].t/second, np.array(spike_monitors[0].i))
    spikes_inh = (spike_monitors[1].t/second, np.array(spike_monitors[1].i))
    
    if state:
        gna_factor = 1 / (1 + 90/105)
        
        exc_currents = ((state_monitors[0].v / mV - 0) * state_monitors[0].ge / uS).mean(axis=0)
        ebg_currents = ((state_monitors[0].v / mV - 0) * state_monitors[0].geb / uS).mean(axis=0)
        ext_currents = ((state_monitors[0].v / mV - 0) * state_monitors[0].gx / uS).mean(axis=0)
        inh_currents = ((state_monitors[0].v / mV + 80) * state_monitors[0].gi / uS).mean(axis=0)
        ibg_currents = ((state_monitors[0].v / mV + 80) * state_monitors[0].gib / uS).mean(axis=0)
        
        exc_I_exc = ((state_monitors[0].v / mV - 90) * gna_factor * state_monitors[0].ge / uS).sum(axis=0)
        ext_I_exc = ((state_monitors[0].v / mV - 90) * gna_factor * state_monitors[0].gx / uS).sum(axis=0)
        ebg_I_exc = ((state_monitors[0].v / mV - 90) * gna_factor * state_monitors[0].geb / uS).sum(axis=0)

        exc_I_inh = ((state_monitors[1].v / mV - 90) * gna_factor * state_monitors[1].ge / uS).sum(axis=0)
        ext_I_inh = ((state_monitors[1].v / mV - 90) * gna_factor * state_monitors[1].gx / uS).sum(axis=0)
        ebg_I_inh = ((state_monitors[1].v / mV - 90) * gna_factor * state_monitors[1].geb / uS).sum(axis=0)
        
        tot_Na_current = (exc_I_exc + ext_I_exc + exc_I_inh + ext_I_inh + ebg_I_exc + ebg_I_inh, state_monitors[0].t/second)
        
        Na_currents = (exc_I_exc, ext_I_exc, exc_I_inh, ext_I_inh, ebg_I_exc, ebg_I_inh, state_monitors[0].t/second)
        
    
    device.delete()
    device.reinit()
    
    if not state:
        return spikes_exc, spikes_inh
    else:
        return spikes_exc, spikes_inh, Na_currents, (exc_currents, ext_currents, inh_currents, ebg_currents, ibg_currents)
    
    # if state:
    #     return spike_monitors, state_monitors
    # else:
    #     return spike_monitors
    

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
    parser.add_argument('--currents', type=int, default=0)
    parser.add_argument('--fixext', type=int, default=0)
    
    args = parser.parse_args()
    
    np.random.seed(seed=args.seed)
    group = args.group
    
    cm_seed = args.seed
    xc = args.corr
    np.random.seed(seed=cm_seed)
    
    if args.fixext > 0:
        cm = connection_matrix(xc=xc, eps=0.2, fix_ext_count=True)
    else:
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
    
    stim_dur = 2.1
    tdi_cost = TimedArray(stims/xc * Hz, dt=stim_dur*second)

    exc_spikes, inh_spikes, Na_currents, syn_currents = run_network(**simulation_kwargs, tdi=tdi_cost, state=True,
                                             tot_time=len(stims)*stim_dur, seed_num=42)
    
    # exc-exc, ext-exc, exc-inh, ext-inh, background-exc, background_inh
    keys = ['exc-exc','ext-exc','exc-inh','ext-inh','bcg-ext','bcg-inh']
    currents_Na = {}
    curr_time = Na_currents[-1]
    
    for curr, name in zip(Na_currents[:-1], keys):
        currents_Na[name] = extract_mean_current(curr, curr_time, stim_dur=stim_dur, nstim=len(stims))
    
    if args.currents > 0:
        keys = ['exc','ext','inh','ext_background','inh_background']
        currents = {}

        for curr, name in zip(syn_currents, keys):
            currents[name] = extract_mean_current(curr, curr_time, stim_dur=stim_dur, nstim=len(stims))
    # currents = extract_mean_current(*Na_current, stim_dur=stim_dur, nstim=len(stims))
    
    def trial_run(seed):
        spikes_exc, spikes_inh = run_network(**simulation_kwargs, tdi=tdi_spikes, state=False,
                                             tot_time=args.repeats*args.stims*1.1, seed_num=seed)

        exc_counts = extract_counts(*spikes_exc, 800, nstim=args.stims, n_repeat=args.repeats)
        inh_counts = extract_counts(*spikes_inh, 200, nstim=args.stims, n_repeat=args.repeats)
        
        return exc_counts, inh_counts

    for exc_counts, inh_counts in tqdm(Pool().uimap(trial_run, range(args.nruns)), total=args.nruns):
        if args.repeats == 1:
            all_trials_exc.append(exc_counts)
            all_trials_inh.append(inh_counts)
        else:
            all_trials_exc.extend(exc_counts)
            all_trials_inh.extend(inh_counts)
        
    
    all_trials_exc = np.array(all_trials_exc)
    all_trials_inh = np.array(all_trials_inh)
    
    if args.currents == 0:
        pickle.dump((stims, all_trials_exc, all_trials_inh, currents_Na),
                    open( f"data/{group}_xc_{xc*100:.0f}_seed{cm_seed}.p", "wb" ) )
    else:
        pickle.dump((stims, all_trials_exc, all_trials_inh, currents_Na, currents),
                    open( f"data/{group}_xc_{xc*100:.0f}_currents_seed{cm_seed}.p", "wb" ) )
        
    
    os.system('rm -rf standalone\ codes/*')
    # print(all_trials.shape)
    # print(f"Done in {wall_time() - start_time:10.3f}")