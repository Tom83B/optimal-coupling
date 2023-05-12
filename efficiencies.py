import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import warnings

from utils import get_info, load_data, get_costs, group2cond, groups


if __name__ == '__main__':
    groups = ['feedforward','strength01','strength02','strength03','strength05',
          'strength10','strength20','strength30','strength50','strength100']

    xc_list = [1, 2, 3, 5, 10, 20, 50, 80, 100]
    
    eff_dict = {}
    res_dict = {}

    for xc in xc_list:
        eff_dict[xc] = {}
        res_dict[xc] = {}
        
        ff_rest = np.array(get_costs(xc, 'feedforward')).sum(axis=0)[0]
        
        for group in tqdm(groups):
            if group == 'feedforward':
                cond = 0
            else:
                cond = float(group.replace("strength","")) / 100

            info_res = get_info(xc, group, exact=True, full_res=True,
                                max_output=40000, eps=1e-4, rest_cost=ff_rest)

            # info_res = get_info(xc, group, exact=True, full_res=True,
            #                     max_output=40000, eps=1e-4, rest_cost=ff_rest)
            
            info, info_cost = info_res['fun'], info_res['expense']
            eff_dict[xc][cond] = info / info_cost
            res_dict[xc][cond] = info_res
            
    pickle.dump((eff_dict, res_dict),
            open( f"data/jk_optimized_restequal_polyfunc.p", "wb" ) )