import numpy as np
from scipy.interpolate import UnivariateSpline
from utils import load_data, get_costs, polyfunc
from sklearn.decomposition import PCA
from tqdm import tqdm
from optimization.jimbokunisawa import optimize
from scipy.stats import norm
from itertools import product
import pickle


groups = ['feedforward','strength01','strength02','strength03','strength05',
          'strength10','strength20','strength30','strength50','strength100']

xc_list = [1, 2, 3, 5, 10, 20, 50, 80, 100]


def get_efficiency(xc, group, nc, n_trials, currents=False, eps=1e-4):
    intensities, exc_counts, inh_counts, na_current, *_ = load_data(xc, group, currents=currents)
    all_counts = np.concatenate([exc_counts, inh_counts], axis=2)[:n_trials,:,:]

    current_cost, ext_cost, exc_cost, inh_cost = get_costs(xc, group, currents=currents)
    tot_cost = (current_cost + ext_cost + exc_cost + inh_cost)*1e-9

    cost_func = UnivariateSpline(intensities, tot_cost, k=1, s=0)

    X = np.swapaxes(all_counts, 0, 1).reshape((30*n_trials,1000))

    pca = PCA(n_components=nc)
    Xhat = pca.fit_transform(X)
    # Xhat = np.sum(X, axis=1, keepdims=True)

    transformed = Xhat.reshape(30,n_trials,pca.n_components_)

    covs = []

    for i in range(30):
        cov = np.atleast_2d(np.cov(transformed[i,:,:].T))
        covs.append(cov)

    covs = np.array(covs)

    cov_func = np.zeros((nc, nc), dtype=object)
    cov_deriv_func = np.zeros((nc, nc), dtype=object)

    w = (all_counts.sum(axis=2).mean(axis=0) / all_counts.mean(axis=2).var(axis=0))**2

    firing_funcs = []
    firing_funcs_deriv = []

    means_all = transformed.mean(axis=1)

    deg = 7

    for i in range(nc):
        means = means_all[:,i]
        # f = polyfunc(intensities, means, deg=deg, w=w)
        # f_deriv = polyfunc(intensities, means, deg=deg, w=w, deriv=True)
        f = UnivariateSpline(intensities, means, k=1, s=0)
        f_deriv = f.derivative()

        firing_funcs.append(f)
        firing_funcs_deriv.append(f_deriv)

        for j in range(i,nc):
            # f = polyfunc(intensities, covs[:,i,j], deg=deg, w=w)
            f = UnivariateSpline(intensities, covs[:,i,j], k=1, s=0)
            cov_func[i][j] = f
            cov_deriv_func[i][j] = f.derivative()

            # f = polyfunc(intensities, covs[:,i,j], deg=deg, deriv=True, w=w)
            # cov_deriv_func[i][j] = f

    def mean_firing(x):
        return np.array([f(x) for f in firing_funcs])

    def mean_deriv(x):
        return np.array([f(x) for f in firing_funcs_deriv])

    def cov_matrix(x):
        matrix = np.zeros((nc,nc, len(x)), dtype=float)

        for i in (range(nc)):
            for j in range(i,nc):
                cov = cov_func[i][j](x)
                matrix[i,j,:] = cov
                matrix[j,i,:] = cov
        return np.array(matrix)

    def cov_matrix_deriv(x):
        matrix = np.zeros((nc,nc,len(x)), dtype=float)

        for i in (range(nc)):
            for j in range(i,nc):
                cov = cov_deriv_func[i][j](x)
                matrix[i,j,:] = cov
                matrix[j,i,:] = cov
        return matrix

    xx = np.linspace(intensities[0], intensities[-1], 1000)
    all_covs = cov_matrix(intensities)
    all_covs_deriv = cov_matrix_deriv(intensities)

    part1_arr = []
    part2_arr = []
    fishinf = []

    for i, x in (enumerate(intensities[:])):
        cov = all_covs[:,:,i]
        dcov = all_covs_deriv[:,:,i]
        inv_cov = np.linalg.inv(cov)

        idcv = inv_cov@dcov

        part1 = mean_deriv(x)[:]@inv_cov@mean_deriv(x)[:]
        part2 = 0.5*np.trace(idcv@idcv)
        # part2 = 0

        part1_arr.append(part1)
        part2_arr.append(part2)

        # fishinf.append(part1)
        fishinf.append(part1 + part2)

    fishinf = np.array(fishinf)
    fishinf_func = UnivariateSpline(intensities, fishinf, k=1, s=0)
    fishinf_all = fishinf_func(xx)

    var = 1 / fishinf_all

    yy = np.linspace((xx - 4 * np.sqrt(var)).min(), (xx + 4 * np.sqrt(var)).max(), 30000)
    # yy = np.arange(30000)

    # fanos = covs.flatten() / means_all[:,0]
    # fano_func = UnivariateSpline(means_all[:,0], fanos, k=1, s=0)
    # fanos_all = fano_func(mean_firing(xx)).flatten()

    # pdfs = norm.pdf(yy, loc=np.atleast_2d(mean_firing(xx)).T, scale=np.atleast_2d(np.sqrt(fanos_all*mean_firing(xx))).T)
    # pdfs = norm.pdf(yy, loc=np.atleast_2d(mean_firing(xx)).T, scale=np.atleast_2d(np.sqrt(all_covs[0])).T)
    pdfs = norm.pdf(yy, loc=np.atleast_2d(xx).T, scale=np.atleast_2d(np.sqrt(var)).T)
    pdfs = (pdfs.T / pdfs.sum(axis=1)).T
    probs = []

    cost_arr = cost_func(xx)

    jk_res = optimize(pdfs, expense=cost_arr, eps=eps)
    return jk_res['fun'] / jk_res['expense']

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=10800)
    
    args = parser.parse_args()
    
    res = {}

    for xc, group in tqdm(product(xc_list, groups), total=len(xc_list)*len(groups)):
        res[(xc, group)] = get_efficiency(xc, group, 500, args.trials)
        
    pickle.dump(res, open( f"efficiencies_{args.trials}trials_nc500_fullcalc.p", "wb" ) )