import numpy as np
from helper import *
from copy import deepcopy
from scipy.optimize import minimize

class LR:
    
    def train_model(self, x, y, x_control, loss_function, max_iter, apply_fairness_constraints, sensitive_attrs, sensitive_attrs_to_cov_thresh):
        max_iter = max_iter

        if apply_fairness_constraints == 0:
            w = minimize(fun = loss_function,
                         x0 = np.random.rand(x.shape[1],),
                         args = (x, y),
                         method = 'SLSQP',
                         options = {"maxiter": max_iter},
                         constraints = [])

        else:
            constraints = self.get_constraint_list_cov(x, y, x_control, sensitive_attrs, sensitive_attrs_to_cov_thresh)
            f_args=(x, y)
            w = minimize(fun = loss_function,
                         x0 = np.random.rand(x.shape[1],),
                         args = f_args,
                         method = 'SLSQP',
                         options = {"maxiter": max_iter},
                         constraints = constraints)

        return w.x

    
    def get_constraint_list_cov(self, x_train, y_train, x_control_train, sensitive_attrs, sensitive_attrs_to_cov_thresh):

        constraints = []
        
        for attr in sensitive_attrs:
            attr_arr = x_control_train[attr]
            attr_arr_transformed, index_dict = get_one_hot_encoding(attr_arr)

            if index_dict is None: # binary attribute
                thresh = sensitive_attrs_to_cov_thresh[attr]
                c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov, 'args':(x_train, y_train, attr_arr_transformed, thresh, False)})
                constraints.append(c)
            else: # otherwise, its a categorical attribute, so we need to set the cov thresh for each value separately
                for attr_val, ind in index_dict.items():
                    attr_name = attr_val
                    thresh = sensitive_attrs_to_cov_thresh[attr][attr_name]

                    t = attr_arr_transformed[:,ind]
                    c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov, 'args':(x_train, y_train, t, thresh, False)})
                    constraints.append(c)

        return constraints