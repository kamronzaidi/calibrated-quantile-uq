import tqdm
import random
import math
import numpy as np
from scipy.interpolate import interp1d
import torch
import matplotlib.pyplot as plt
from cali_plot import get_props, plot_calibration_curve, ens_get_props
from sklearn.preprocessing import KBinsDiscretizer
from numpy import histogramdd

from argparse import Namespace
import sys

# sys.path.append("/home/master/rs/uq/toolkit/uq_toolkit")
# from quantile_models import (
#     crps_score,
#     bag_nll,
#     check_loss,
#     interval_score,
#     get_obs_props,
# )
import uncertainty_toolbox as uct


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_q_idx(exp_props, q):
    target_idx = None
    for idx, x in enumerate(exp_props):
        if idx + 1 == exp_props.shape[0]:
            if round(q, 2) == round(float(exp_props[-1]), 2):
                target_idx = exp_props.shape[0] - 1
            break
        if x <= q < exp_props[idx + 1]:
            target_idx = idx
            break
    if target_idx is None:
        import pdb

        pdb.set_trace()
        raise ValueError("q must be within exp_props")
    return target_idx


def test_uq(
    model,
    x,
    y,
    exp_props,
    y_range,
    recal_model=None,
    recal_type=None,
    make_plots=False,
    test_group_cal=False,
):

    # obs_props, quantile_preds, quantile_preds_mat = \
    #     ens_get_props(model, x, y, exp_props=exp_props, recal_model=recal_model,
    #                   recal_type=recal_type)

    num_pts = x.shape[0]
    y = y.detach().cpu().reshape(num_pts, -1)

    quantile_preds = model.predict_q(
        x,
        exp_props,
        ens_pred_type="conf",
        recal_model=recal_model,
        recal_type=recal_type,
    )  # of shape (num_pts, num_q)
    obs_props = torch.mean((quantile_preds >= y).float(), dim=0).flatten()

    assert exp_props.shape == obs_props.shape

    idx_01 = get_q_idx(exp_props, 0.01)
    idx_99 = get_q_idx(exp_props, 0.99)
    cali_score = plot_calibration_curve(
        exp_props[idx_01 : idx_99 + 1],
        obs_props[idx_01 : idx_99 + 1],
        make_plots=make_plots,
    )

    order = torch.argsort(y.flatten())
    q_025 = quantile_preds[:, get_q_idx(exp_props, 0.025)][order]
    q_975 = quantile_preds[:, get_q_idx(exp_props, 0.975)][order]
    sharp_score = torch.mean(q_975 - q_025).item() / y_range

    if make_plots:
        plt.figure(figsize=(8, 5))
        plt.plot(torch.arange(y.size(0)), q_025)
        plt.plot(torch.arange(y.size(0)), q_975)
        plt.scatter(torch.arange(y.size(0)), y[order], c="r")
        plt.title("Mean Width: {:.3f}".format(sharp_score))
        plt.show()

    g_cali_scores = []
    if test_group_cal:
        ratio_arr = np.linspace(0.01, 1.0, 10)
        print(
            "Spanning group size from {} to {} in {} increments".format(
                np.min(ratio_arr), np.max(ratio_arr), len(ratio_arr)
            )
        )
        for r in tqdm.tqdm(ratio_arr):
            gc = test_group_cali(
                y=y,
                q_pred_mat=quantile_preds[:, idx_01 : idx_99 + 1],
                exp_props=exp_props[idx_01 : idx_99 + 1],
                y_range=y_range,
                ratio=r,
            )
            g_cali_scores.append(gc)
        g_cali_scores = np.array(g_cali_scores)

        # plt.plot(ratio_arr, g_cali_scores)
        # plt.show()

    """ Get some scoring rules """
    args = Namespace(
        device=torch.device("cpu"), q_list=torch.linspace(0.01, 0.99, 99)
    )
    # curr_crps = crps_score(model, x, y, args)
    # curr_nll = bag_nll(model, x, y, args)
    # curr_check = check_loss(model, x, y, args)
    # curr_int = interval_score(model, x, y, args)
    # int_exp_props, int_obs_props, int_cdf_preds = get_obs_props(#0.95 PI
    #     model, x, y, exp_props=None, device=args.device, type="interval"
    # )
    # curr_int_cali = torch.mean(torch.abs(int_exp_props - int_obs_props)).item()

    # curr_scoring_rules = {
    #     "crps": float(curr_crps),
    #     "nll": float(curr_nll),
    #     "check": float(curr_check),
    #     "int": float(curr_int),
    #     "int_cali": float(curr_int_cali),
    # }
    curr_scoring_rules = {
        "crps": 0,
        "nll": 0,
        "check": 0,
        "int": 0,
        "int_cali": 0,
    }

    #pnn_metrics = uct.metrics.get_all_metrics(pred_mean, pred_std, te_y)
    return (
        cali_score,
        sharp_score,
        obs_props,
        quantile_preds,
        g_cali_scores,
        curr_scoring_rules,
    )


def test_group_cali(
    y,
    q_pred_mat,
    exp_props,
    y_range,
    ratio,
    num_group_draws=20,
    make_plots=False,
):

    num_pts, num_q = q_pred_mat.shape
    group_size = max([int(round(num_pts * ratio)), 2])
    q_025_idx = get_q_idx(exp_props, 0.025)
    q_975_idx = get_q_idx(exp_props, 0.975)

    # group_obs_props = []
    # group_sharp_scores = []
    score_per_trial = []
    for _ in range(20):  # each trial
        ##########
        group_cali_scores = []
        for g_idx in range(num_group_draws):
            rand_idx = np.random.choice(num_pts, group_size, replace=True)
            g_y = y[rand_idx]
            g_q_preds = q_pred_mat[rand_idx, :]
            g_obs_props = torch.mean(
                (g_q_preds >= g_y).float(), dim=0
            ).flatten()
            assert exp_props.shape == g_obs_props.shape
            g_cali_score = plot_calibration_curve(
                exp_props, g_obs_props, make_plots=False
            )
            # g_sharp_score = torch.mean(
            #     g_q_preds[:,q_975_idx] - g_q_preds[:,q_025_idx]
            # ).item() / y_range

            group_cali_scores.append(g_cali_score)
            # group_obs_props.append(g_obs_props)
            # group_sharp_scores.append(g_sharp_score)

        # mean_cali_score = np.mean(group_cali_scores)
        mean_cali_score = np.max(group_cali_scores)
        ##########

        score_per_trial.append(mean_cali_score)

    return np.mean(score_per_trial)

    # mean_sharp_score = np.mean(group_sharp_scores)
    # mean_group_obs_props = torch.mean(torch.stack(group_obs_props, dim=0), dim=0)
    # mean_group_cali_score = plot_calibration_curve(exp_props, mean_group_obs_props,
    #                                                make_plots=False)

    return mean_cali_score


def gather_loss_per_q(loss_fn, model, y, x, q_list, device, args):
    """
    Evaluate loss_fn for eqch q in q_listKBinsDiscretizer
    loss_fn must only take in a scalar q
    """
    loss_list = []
    for q in q_list:
        q_loss = loss_fn(model, y, x, q, device, args)
        loss_list.append(q_loss)
    loss = torch.mean(torch.stack(loss_list))

    return loss


def discretize_domain(x_arr, batch_size):
    num_pts, dim_x = x_arr.shape

    group_list = []
    for d in range(dim_x):
        dim_order = np.argsort(x_arr[:, d]).flatten()
        curr_group = [
            dim_order[i : i + batch_size] for i in range(0, num_pts, batch_size)
        ]
        assert len(curr_group) == math.ceil(num_pts / batch_size)
        group_list.append(curr_group)
    return group_list


def discretize_domain_old(x_arr, min_pts):
    num_pts, dim_x = x_arr.shape
    # n_bins = 2 * np.ones(dim_x).astype(int)

    group_data_idxs = []
    while len(group_data_idxs) < 1:
        n_bins = np.random.randint(low=1, high=3, size=dim_x)
        H, edges = histogramdd(x_arr, bins=n_bins)

        group_idxs = np.where(H >= min_pts)
        group_bounds = []
        for g_idx in zip(*group_idxs):
            group_bounds.append(
                [(edges[i][x], edges[i][x + 1]) for i, x in enumerate(g_idx)]
            )

        for b_list in group_bounds:
            good_dim_idxs = []
            for d_idx, (l, u) in enumerate(b_list):
                good_dim_idxs.append(
                    (l <= x_arr[:, d_idx]) * (x_arr[:, d_idx] < u)
                )
            curr_group_1 = np.prod(np.stack(good_dim_idxs, axis=0), axis=0)
            curr_group_idx = np.where(curr_group_1.flatten() > 0)
            if curr_group_idx[0].size < min_pts:
                continue
            group_data_idxs.append(curr_group_idx[0])

    rand_dim_idx = np.random.randint(dim_x)
    group_pts_idx = list(np.concatenate(group_data_idxs))

    if num_pts - len(group_pts_idx) >= (min_pts // 2):
        rest_rand_sorted = list(np.argsort(x_arr[:, rand_dim_idx]))
        for item in group_pts_idx:
            rest_rand_sorted.remove(item)
        rest_group_size = int(min_pts // 2)
        beg_idx = 0
        rest_group_data_idxs = []
        while beg_idx < len(rest_rand_sorted):
            if beg_idx + rest_group_size >= len(rest_rand_sorted):
                end_idx = len(rest_rand_sorted)
            else:
                end_idx = beg_idx + rest_group_size

            if (
                np.array(rest_rand_sorted[beg_idx:end_idx]).size
                >= rest_group_size
            ):
                rest_group_data_idxs.append(
                    np.array(rest_rand_sorted[beg_idx:end_idx])
                )
            beg_idx = end_idx
        group_data_idxs.extend(rest_group_data_idxs)

    assert np.array([x.size > 0 for x in group_data_idxs]).all()

    return group_data_idxs


if __name__ == "__main__":
    temp_x = np.random.uniform(0, 100, size=[100, 2])
    # num_bins, idxs = discretize_domain(temp_x)

    group_idxs = discretize_domain(temp_x, 30)
    cum_num_pts = 0
    for i in group_idxs:
        g = temp_x[i.flatten()]
        print(g.shape)
        cum_num_pts += g.shape[0]
        plt.plot(g[:, 0], g[:, 1], "o")
    print(cum_num_pts)
    plt.show()

    # for i in idxs:
    #     g = temp_x[i.flatten()]
    #     print(g.shape)
    #     cum_num_pts += g.shape[0]
    #     plt.plot(g[:,0], g[:,1], '^')
    # plt.show()
    # assert cum_num_pts == temp_x.shape[0]
    # for i in range(5):
    #     for j in range(5):
    #
    # for i in range(num_bins):
    #     plt.plot(temp_x[idxs[i], 0], temp_x[idxs[i], 1], 'o')
    # plt.show()
