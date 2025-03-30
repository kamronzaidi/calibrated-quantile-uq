import torch
from sklearn.isotonic import IsotonicRegression
from utils.misc_utils import get_q_idx
from auxiliary.interp1d import *
import numpy as np

def get_cdf1_cdf2(model, X, Y1, Y2 = None, num_samples = 10000):
    # COMPUTE CDF FOR SAME VALUE OF X AND TWO DIFFERENT VALUES OF Y. REQUIRED TO COMPUTE
    # FINITE DIFFERENCE USED TO COMPUTE LIKELIHOOD. SIMULTANEOUS COMPUTATION FOR DIFFERENT YS
    # NECESSARY DUE TO SAMPLING-BASED COMPUTATION

    length_data = X.shape[0]
    q_levels = torch.linspace(0, 1, num_samples).to(model.device)
    
    quantiles = model.predict_q(
        X,
        q_levels,
    )#model(X, quantile=q_levels)

    F1 = interp1d(quantiles, q_levels, Y1).clamp(min=0, max=1)
    if Y2 is None:
        return F1
    else:
        F2 = interp1d(quantiles, q_levels, Y2).clamp(min=0, max=1)
        return F1, F2

def iso_recal_ours(model, X, Y):
    cdf_vals = get_cdf1_cdf2(model, X, Y)
    Phat_vals = []
    for cdf in cdf_vals:
        cdf_l = cdf_vals <= cdf
        cdf_ind = torch.zeros(cdf_vals.shape)
        cdf_ind[cdf_l] = 1
        Phat_vals.append(torch.mean(cdf_ind))

    Phat_vals = torch.stack(Phat_vals).reshape(cdf_vals.shape[0], 1)

    ## CONVERT Phat_Vals and cdf_vals TO NUMPY FOR ISOTONIC REGRESSION
    cdf_vals_np = np.float64(cdf_vals.detach().cpu().numpy())
    Phat_vals_np = np.float64(Phat_vals.cpu().numpy().reshape(Phat_vals.shape[0], ))

    iso_reg = IsotonicRegression(out_of_bounds='clip').fit(cdf_vals_np, Phat_vals_np)
    return iso_reg

def iso_recal(exp_props, obs_props):
    exp_props = exp_props.flatten()
    obs_props = obs_props.flatten()
    min_obs = torch.min(obs_props)
    max_obs = torch.max(obs_props)

    iso_model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    try:
        assert torch.min(obs_props) == 0.0
        assert torch.max(obs_props) == 1.0
    except:
        print("Obs props not ideal: from {} to {}".format(min_obs, max_obs))
    # just need observed prop values between 0 and 1
    # problematic if min_obs_p > 0 and max_obs_p < 1

    exp_0_idx = get_q_idx(exp_props, 0.0)
    exp_1_idx = get_q_idx(exp_props, 1.0)
    within_01 = obs_props[exp_0_idx : exp_1_idx + 1]

    beg_idx, end_idx = None, None
    # handle beg_idx
    min_obs_below = torch.min(obs_props[:exp_0_idx])
    min_obs_within = torch.min(within_01)
    if min_obs_below < min_obs_within:
        i = exp_0_idx - 1
        while obs_props[i] > min_obs_below:
            i -= 1
        beg_idx = i
    elif torch.sum((within_01 == min_obs_within).float()) > 1:
        # multiple minima in within_01 ==> get last min idx
        i = exp_1_idx - 1
        while obs_props[i] > min_obs_within:
            i -= 1
        beg_idx = i
    elif torch.sum((within_01 == min_obs_within).float()) == 1:
        beg_idx = torch.argmin(within_01) + exp_0_idx
    else:
        import pudb

        pudb.set_trace()

    # handle end_idx
    max_obs_above = torch.max(obs_props[exp_1_idx + 1 :])
    max_obs_within = torch.max(within_01)
    if max_obs_above > max_obs_within:
        i = exp_1_idx + 1
        while obs_props[i] < max_obs_above:
            i += 1
        end_idx = i + 1
    elif torch.sum((within_01 == max_obs_within).float()) > 1:
        # multiple minima in within_01 ==> get last min idx
        i = beg_idx
        while obs_props[i] < max_obs_within:
            i += 1
        end_idx = i + 1
    elif torch.sum((within_01 == max_obs_within).float()) == 1:
        end_idx = exp_0_idx + torch.argmax(within_01) + 1
    else:
        import pudb

        pudb.set_trace()

    assert end_idx > beg_idx

    # min_idx = torch.argmin(obs_props)
    # last_idx = obs_props.size(0)
    # beg_idx, end_idx = None, None
    #
    # for idx in range(min_idx, last_idx):
    #     if obs_props[idx] >= 0.0:
    #         if beg_idx is None:
    #             beg_idx = idx
    #     if obs_props[idx] >= 1.0:
    #         if end_idx is None:
    #             end_idx = idx+1
    # # if beg_idx is None:
    # #     beg_idx = 0
    # # if end_idx is None:
    # #     end_idx = obs_props.size(0)
    # print(beg_idx, end_idx)

    filtered_obs_props = obs_props[beg_idx:end_idx]
    filtered_exp_props = exp_props[beg_idx:end_idx]

    try:
        iso_model = iso_model.fit(filtered_obs_props, filtered_exp_props)
    except:
        import pudb

        pudb.set_trace()

    return iso_model


if __name__ == "__main__":
    exp = torch.linspace(-0.5, 1.5, 200)
    from copy import deepcopy

    obs = deepcopy(exp)
    obs[:80] = 0
    obs[-80:] = 1

    iso_recal(exp, obs)
