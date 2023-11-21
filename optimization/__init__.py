import numpy as np
import warnings


def sensitivity(sr_grid, prior):
    # prior = stimulus_pdf.dot(sr_grid)
    top_lim = 1e2
    specificity = np.ones_like(sr_grid) * np.inf
#     specificity[:,prior > 0] = sr_grid[:,prior > 0] / prior[prior > 0]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'invalid value encountered in divide')
        warnings.filterwarnings('ignore', r'divide by zero encountered in log2')
        warnings.filterwarnings('ignore', r'divide by zero encountered in divide')
        specificity = sr_grid / prior
        
        return np.nan_to_num((sr_grid * np.nan_to_num(np.clip(np.log2(specificity), a_min=-1000, a_max=1000))).sum(axis=1))

# def sensitivity_func(sr_func, prior):
#     def specificity(x):
#         spec = sr_func(x) / prior
#         spec[np.isnan(spec)] = 1
#         return spec
#     specificity = lambda x: sr_func(x) / prior

def information(sr_grid, stimulus_pdf):
    posterior = stimulus_pdf.dot(sr_grid)
    return sensitivity(sr_grid, posterior).dot(stimulus_pdf)