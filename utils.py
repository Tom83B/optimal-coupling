import numpy as np
from brian2.units import *
from scipy.stats import lognorm
from scipy.interpolate import UnivariateSpline
from optimization import slsqpopt, cuttingplane, blahut_arimoto, jimbokunisawa
import pickle
import warnings
import os
from matplotlib.pyplot import get_cmap

ELEMENTARY_CHARGE = 1.6e-19
AP_COST = 0.125e9*5

groups = ['feedforward','strength01','strength02','strength03','strength05','strength10',
              'strength20','strength30','strength50','strength100']

cmap = get_cmap('viridis')
colors = np.linspace(0, 1, len(groups))
color_dict = {group: cmap(c) for c, group in zip(colors, groups)}


def despine_ax(ax, where=None, remove_ticks=None):
    if where is None:
        where = 'trlb'
    if remove_ticks is None:
        remove_ticks = where
    
    if remove_ticks is not None:
        if 'b' in where:
            ax.set_xticks([])
            ax.set_xticklabels([])
        if 'l' in where:
            ax.set_yticks([])
            ax.set_yticklabels([])
    
    to_despine = []
    
    if 'r' in where:
        to_despine.append('right')
    if 't' in where:
        to_despine.append('top')
    if 'l' in where:
        to_despine.append('left')
    if 'b' in where:
        to_despine.append('bottom')
    
    for side in to_despine:
        ax.spines[side].set_visible(False)

def extract_mean_current(current, time, nstim=30, stim_dur=1.1, discard=0.1):
    mean_currents = []
    
    for s in range(nstim):
        mask = (time > discard+s*stim_dur) & (time <= (s+stim_dur-discard)*stim_dur)
        mean_currents.append(current[mask].mean())
    
    return np.array(mean_currents)

def group2cond(group):
    if group == 'feedforward':
        cond = 0
    else:
        cond = float(group.replace("strength","")) / 100
        
    return cond

def connection_matrix(eps=0.2, xc=0.2, fix_ext_count=False, N_x=1000, N_exc=800, N_inh=200, rand_matrix=None):
    N_tot = N_x + N_exc + N_inh
    
    if rand_matrix is None:
        rand_matrix = np.random.rand(N_tot, N_tot)
    
    conn_matrix = np.zeros(shape=(N_tot, N_tot))
    
    
    conn_matrix[N_x:,N_x:] = rand_matrix[N_x:,N_x:] < eps
    
    if xc == 0:
        conn_matrix[:N_x,N_x:] = np.eye(N_x, N_exc+N_inh)
    else:
        if fix_ext_count:
            n = int(xc * N_x)
            indices = np.argpartition(rand_matrix[:N_x,N_x:], -n, axis=0)[-n:].T

            for ii in range(N_exc+N_inh):
                conn_matrix[:N_x,N_x:][:,ii][indices[ii]] = 1
        else:
            conn_matrix[:N_x,N_x:] = rand_matrix[:N_x,N_x:] < xc
    
    return conn_matrix

def connection_matrix_RS_FS(p_RS_RS=0.15, p_RS_FS=0.4, p_FS=0.5, xc=0.2, fix_syn_count=False, N_x=1000, N_exc=800, N_inh=200, rand_matrix=None):
    """rows are sources, columns are targets"""
    N_tot = N_x + N_exc + N_inh
    
    if rand_matrix is None:
        rand_matrix = np.random.rand(N_tot, N_tot)
    
    conn_matrix = np.zeros(shape=(N_tot, N_tot))
    
    
    conn_matrix[N_x:N_x+N_exc,N_x:N_x+N_exc] = rand_matrix[N_x:N_x+N_exc,N_x:N_x+N_exc] < p_RS_RS
    conn_matrix[N_x:N_x+N_exc,N_x+N_exc:] = rand_matrix[N_x:N_x+N_exc,N_x+N_exc:] < p_RS_FS
    conn_matrix[N_x+N_exc:,N_x:] = rand_matrix[N_x+N_exc:,N_x:] < p_FS
    
    if xc == 0:
        conn_matrix[:N_x,N_x:] = np.eye(N_x, N_exc+N_inh)
    else:
        conn_matrix[:N_x,N_x:] = rand_matrix[:N_x,N_x:] < xc
    
    return conn_matrix

# def extract_counts(monitor, N):
#     spiketrain = monitor.t / second
#     indices = np.array(monitor.i)
    
#     mask = (spiketrain > 1) & (spiketrain <= 2)
#     ixs, counts = np.unique(indices[mask], return_counts=True)
#     counts_all = np.zeros(N)
#     nonzero_mask = np.isin(np.arange(N), ixs)
#     counts_all[nonzero_mask] = counts
        
#     return counts_all

def fisher_prob(xx, yy, dyx, fanos, cost_func=None, lamw=0):
    F = dyx/(yy*fanos)
    
    if cost_func is not None:
        prob = np.sqrt(F)*np.exp(-lamw*cost_func(xx))
    else:
        prob = np.sqrt(F)
    
    return prob / prob.sum()

def fisher_lower_bound(xx, yy, dyx, fanos, cost_func=None, lamw=0):
    F = dyx/(yy*fanos)
    
    if cost_func is not None:
        prob = np.sqrt(F)*np.exp(-lamw*cost_func(xx))
    else:
        prob = np.sqrt(F)
    
    prob = prob / prob.sum()
    return -prob @ np.log2(prob * np.sqrt(2*np.pi*np.e / F)), prob@cost_func(xx)
    

def get_sr_grid(xx_input, mean_func, fffunc, min_output=0, max_output=40000):
    outputs = np.arange(max_output)
    fyx = []
    
    mask = (mean_func(xx_input) >= min_output)
    xx_input = xx_input[mask]

    for x, mean, fano in zip(xx_input, mean_func(xx_input), fffunc(xx_input)):
        var = fano*mean

        scale = mean / np.sqrt(1+var/(mean**2))

        if mean < 0:
            print(mean)
            break

        s = np.sqrt(np.log(1+var/(mean**2)))
        probs = lognorm.pdf(outputs, s=s, scale=scale)*(outputs[1]-outputs[0])
        probs = probs / probs.sum()
        
        if probs.sum() != probs.sum():
            import pdb; pdb.set_trace()
            
        fyx.append(probs)
        
    fyx = np.array(fyx)
    return fyx


def group_color(group):
    """returns a color from a viridis colormap"""
    
    
def polyfunc(xx, yy, deg, deriv=False, **kwargs):
    coefs = np.polyfit(xx, yy, deg=deg, **kwargs)
    
    xx_new = np.linspace(xx[0], xx[-1], 1000)
    
    yy_fit = np.zeros_like(xx_new)
    
    if deriv is False:
        for i, coef in enumerate(coefs):
            yy_fit += coef * xx_new**(deg-i)
    else:
        for i, coef in enumerate(coefs[:-1]):
            yy_fit += (deg-i) * coef * xx_new**(deg-i-1)
        
    return UnivariateSpline(xx_new, yy_fit, s=0, k=1)
    

def information(input_prob, xx_input, mean_func, fffunc, cost_func, min_output=0, max_output=1e6):
# #     prob = np.sqrt(F)*np.exp(-lamw*f(np.log(xx_input)))
# #     outputs = np.linspace(0, 25*800, 1000)
#     outputs = np.arange(25*800)
#     # qy = np.zeros_like(outputs)
#     fyx = []
    
    mask = (mean_func(xx_input) >= min_output) & (mean_func(xx_input) < max_output)
#     xx_input = xx_input[mask]

#     for x, mean, fano in zip(xx_input, mean_func(xx_input), fffunc(mean_func(xx_input))):
#         var = fano*mean

#         scale = mean / np.sqrt(1+var/(mean**2))

#         if mean < 0:
#             print(mean)
#             break

#         s = np.sqrt(np.log(1+var/(mean**2)))
#     #     print(var / mean)
#     #     if var / mean > 1:
#     #         break
#     #     qy += p*lognorm.pdf(outputs, s=s, scale=scale)
#         fyx.append(lognorm.pdf(outputs, s=s, scale=scale)*(outputs[1]-outputs[0]))
        
    fyx = get_sr_grid(xx_input, mean_func, fffunc, min_output=min_output)
    
    if len(input_prob.shape) == 1:
        input_prob = input_prob[mask]
        input_prob = input_prob / input_prob.sum()
        qy = fyx.T@input_prob

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in divide')
            warnings.filterwarnings('ignore', r'divide by zero encountered in divide')
            specificity = fyx / qy
        info = input_prob @ np.nan_to_num((fyx * np.nan_to_num(np.clip(np.log2(specificity), a_min=-1000, a_max=1000))).sum(axis=1))
        cost = input_prob @ cost_func(xx_input)
    
        return info, cost
    
    else:
        costs = []
        infos = []
        for ip in input_prob:
            ip = ip[mask]
            ip = ip / ip.sum()
            qy = fyx.T@ip


            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'invalid value encountered in divide')
                warnings.filterwarnings('ignore', r'divide by zero encountered in log2')
                warnings.filterwarnings('ignore', r'divide by zero encountered in divide')
                specificity = fyx / qy
                info = ip @ np.nan_to_num((fyx * np.nan_to_num(np.clip(np.log2(specificity), a_min=-1000, a_max=1000))).sum(axis=1))
                cost = ip @ cost_func(xx_input)
            
            infos.append(info)
            costs.append(cost)
        
        return infos, costs
    
def load_currents(xc, group):
    """
    Load the calculated Na current for given xc (in percent) and group
    returns array of length corresponding to the number of different intensities tested
    """
    groups = ['feedforward','strength01','strength03','strength05','strength07','strength10','strength15',
              'strength20','strength30','strength50','strength100','strength200']
    
    ix_dict = {group: ix for ix, group in enumerate(groups)}
    
    with open(f'data standalone/currents/Na_currents_xc{xc}.p', "rb") as input_file:
        if xc not in [1,2,3]:
            currents = pickle.load(input_file)[xc][ix_dict[group]]
        else:
            currents = pickle.load(input_file)[group2cond(group)]
        
    return currents

def load_efficiencies(filename):
    """
    returns dicts:
        with efficiencies (key1: xc, key2: conductance)
        with results of JK optimization
    """
    with open(filename, "rb") as input_file:
        eff_dict, res_dict = pickle.load(input_file)
        
    return eff_dict, res_dict

def load_data(xc, group, currents=False):
    """
    returns as a tuple:
    1. intensities used for simulation (array of lenght n_intensities)
    2. excitatory AP counts (array of shape (n_trials, n_intensities, n_exc_neurons))
    3. inhibitory AP counts (array of shape (n_trials, n_intensities, n_inh_neurons))
    4. the total Na current
    (5. synaptic currents if currents is True)
    """
    
    if currents is False:
        with open(f'data/{group}_xc_{xc:.0f}_seed0.p', "rb") as input_file:
            intensities, exc_counts, inh_counts, na_current = pickle.load(input_file)
        
            return intensities, exc_counts, inh_counts, na_current
    else:
        with open(f'data/{group}_xc_{xc:.0f}_currents_seed0.p', "rb") as input_file:
            intensities, exc_counts, inh_counts, na_current, syn_currents = pickle.load(input_file)
        
            return intensities, exc_counts, inh_counts, na_current, syn_currents

def get_costs(xc, group, currents=None):
    """
    takes xc (in percents) and group (string) and returns the costs as a tuple in the following order:
    1. cost of excitatory currents
    2. cost of spikes of external population
    3. cost of excitatory recurrent spikes
    4. cost of inhibitory recurrent spikes
    """
    intensities, exc_counts, inh_counts, currents = load_data(xc, group)
    
    keys = ['exc-exc','ext-exc','exc-inh','ext-inh','bcg-ext','bcg-inh']
    coefs = [1, 1, 1, 1, 1, 1]
    tot_currents = np.zeros_like(currents[keys[0]])
    
    for key, coef in zip(keys, coefs):
        tot_currents += coef*currents[key]
    # currents = extract_mean_current(*na_current, stim_dur=2.1)
        
    current_cost = -tot_currents / (3*ELEMENTARY_CHARGE) / 1e9
    
    ext_spikes = 100 * 1000 * intensities / xc
    exc_spikes = exc_counts.mean(axis=0).sum(axis=1)
    inh_spikes = inh_counts.mean(axis=0).sum(axis=1)
    
    # spike_costs = (ext_spikes + exc_spikes + inh_spikes)*AP_COST
    
    return current_cost,  ext_spikes*AP_COST, exc_spikes*AP_COST, inh_spikes*AP_COST

def get_info(xc, group, divisive=0, currents=None, exact=False, max_output=30000, full_res=False, rest_cost=None,
            min_output=0, cost_pair=None, eps=1e-2):
    intensities, exc_counts, inh_counts, na_current = load_data(xc, group)

    ext_spikes = 100 * 1000 * intensities / xc
    exc_spikes = exc_counts.mean(axis=0).sum(axis=1)
    inh_spikes = inh_counts.mean(axis=0).sum(axis=1)
    
    tot_counts = exc_counts.sum(axis=2) + inh_counts.sum(axis=2)
    # mean_currents = extract_mean_current(*na_current, stim_dur=2.1)
    # tot_counts = (exc_counts.sum(axis=2) / (1+ divisive * inh_counts.sum(axis=2)))
    
#     if currents is None:
#         currents = load_currents(xc, group)
        
#     current_cost = -currents / (3*ELEMENTARY_CHARGE) / 1e9
    
#     # if group != 'feedforward':
#     #     spike_costs = (ext_spikes + exc_spikes + inh_spikes) * AP_COST
#     # else:
#     spike_costs = (ext_spikes + exc_spikes + inh_spikes)*AP_COST
#     # current_cost = current_cost * 0.8
    
    if cost_pair is None:
        cost_pair = (xc, group)
    
    current_cost, ext_cost, exc_cost, inh_cost = get_costs(*cost_pair, currents)
    costs = (current_cost + ext_cost + exc_cost + inh_cost) * 1e-9
    
    if rest_cost is not None:
        costs = costs - costs.min() + rest_cost*1e-9
    
    # cost_func = UnivariateSpline(tot_counts.mean(axis=0)[last_zero:], costs[last_zero:], k=1, s=0, ext=3)
    cost_func = UnivariateSpline(intensities, costs, k=1, s=0, ext=3)

    # tot_counts = exc_counts.sum(axis=2)
    fanos = tot_counts.var(axis=0) / tot_counts.mean(axis=0)
    
    w = 1 / fanos ** 2
    
    means = tot_counts.mean(axis=0)

    f = UnivariateSpline(intensities, means, k=1, s=0, ext=3)
    xx = np.linspace(intensities[0], intensities[-1], 1000)
    dx = xx[1] - xx[0]
    yy = f(xx)
    yy_func = UnivariateSpline(xx, f(xx), k=1, s=0)
    dyx = f(xx) / dx

    # fffunc = UnivariateSpline(tot_counts.mean(axis=0)[last_zero:], savgol_filter(fanos, window_length=3, polyorder=2)[last_zero:], k=1, s=0, ext=3)
    # fffunc = UnivariateSpline(intensities, fanos, k=1, s=0, ext=3)
    fano_func = polyfunc(means, fanos, 7, w=w)
    fffunc = lambda x: fano_func(f(x))

    if exact is False:
        ips = []
        for lamw in np.logspace(-5, -1, 30):
            input_prob = fisher_prob(xx, yy, dyx, fffunc(yy), cost_func, lamw=lamw)
            ips.append(input_prob)

        infos, info_costs = information(np.array(ips), xx, yy_func, fffunc, cost_func, min_output=min_output)

        return np.array([infos, info_costs])
    else:
        sr_grid = get_sr_grid(xx, yy_func, fffunc, min_output=min_output, max_output=max_output)
        info_res = jimbokunisawa.optimize(sr_grid, eps=eps, verbose=True,
                                          expense=cost_func(xx)[len(xx)-sr_grid.shape[0]:])
        
        if not full_res:
            return np.array([info_res['fun'], info_res['expense']])
        else:
            return info_res