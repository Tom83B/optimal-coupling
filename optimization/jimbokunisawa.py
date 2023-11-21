import numpy as np
import warnings
from . import sensitivity


def optimize(sr_grid, init=None, eps=1e-4, max_iter=100000, verbose=False, expense_factor=1, basal_expense=0, expense=None):
    n = sr_grid.shape[0]  # TODO sjednotit uvodni cast kodu s cutting plane
    
    if expense is None:
        expense = sr_grid.dot(np.arange(sr_grid.shape[1])) * expense_factor + basal_expense

    a = np.min(expense)

    if init is None:
        new_pdf = np.ones(n)
        new_pdf = new_pdf / new_pdf.sum()
    elif all(init >= 0) and np.isclose(np.sum(init), 1.):
        if len(init) == n:
            new_pdf = init
        else:
            raise ValueError('init has incorrect size. {} expected but len(init) is {}'.format(n, len(init)))
    else:
        raise ValueError('init is not a probability distribution')

    old_capacity = -999

    for i in range(max_iter):
        prior = new_pdf.dot(sr_grid)
        kl_div = sensitivity(sr_grid, prior)
        capacity = new_pdf.dot(kl_div)

        accuracy = np.abs(capacity - old_capacity)

        if accuracy < eps:
            break

        p_tilde = new_pdf * np.exp(a * sensitivity(sr_grid, prior) / expense)
        new_pdf = p_tilde / p_tilde.sum()

        old_capacity = capacity
    else:
        warnings.warn(f'Maximum number of iterations ({max_iter}) reached', RuntimeWarning)

    fin_exp = new_pdf.dot(expense)
    out_pdf = sr_grid.T.dot(new_pdf)


    return {'pdf': new_pdf, 'fun': capacity, 'out_pdf': out_pdf, 'sensitivity': sensitivity(sr_grid, out_pdf),
            'accuracy': accuracy, 'expense': fin_exp, 'spike_count': out_pdf.dot(np.arange(out_pdf.shape[0]))}