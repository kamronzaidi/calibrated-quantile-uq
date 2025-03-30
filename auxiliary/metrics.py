import numpy as np
import torch
import torch.nn as nn
import pickle

def check_score(X_test, Y_test, model, p_diagnostics =torch.linspace(0, 1, 25)):

        output_device = model.output_device
        CS = torch.zeros(p_diagnostics.shape).to(output_device)
        quantile_tau = model.get_quantiles(X_test, p_diagnostics)
        diff = Y_test - quantile_tau
        for i in range(p_diagnostics.shape[0]):
            CS[i] = torch.mean(torch.maximum(p_diagnostics[i] * (diff.T[i]), (1 - p_diagnostics[i]) * (-diff.T[i]))).to(output_device)
        return CS.mean()

def expected_calibration_error(X_test, Y_test, model, p_diagnostics = torch.linspace(0, 1, 100)):

    output_device = model.output_device
    p_diagnostics = p_diagnostics.to(model.output_device)
    cdf_vals = model.get_cdf_value(X=X_test, Y=Y_test)
    phat = []
    for pj in p_diagnostics:
        cdf_l = cdf_vals <= pj
        cdf_ind = torch.zeros(cdf_vals.shape).to(output_device)
        cdf_ind[cdf_l] = 1
        phat.append(torch.mean(cdf_ind))
    phat = torch.stack(phat).reshape(p_diagnostics.shape[0],)
    dp = p_diagnostics[1:]- p_diagnostics[:-1]
    err_p = 0.5*(phat - p_diagnostics)[1:] + 0.5*(phat - p_diagnostics)[:-1]
    return torch.sum(err_p**2*dp) # *dp


def negative_log_likelihood(X_test, Y_test, model, dy = 1e-2):

    output_device = model.output_device
    dFdY = model.get_likelihood(X=X_test, Y=Y_test)

    return -torch.mean(torch.log(dFdY).to(output_device)).to(output_device)



def average_variance(X_test, model, p_diagnostics = torch.linspace(0, 1, 100)):

    output_device = model.output_device
    quantiles = model.get_quantiles(X=X_test, q_levels=p_diagnostics)
    dp = (p_diagnostics[1:] - p_diagnostics[:-1]).to(output_device)
    x = 0.5*(quantiles.T[1:] + quantiles.T[:-1])
    Ex = torch.sum(x.T*dp,1)
    varx = torch.sum(((x - Ex)**2).T*dp,1).to(output_device)

    return torch.mean(varx)

def average_95_interval(X_test, model):

    output_device = model.output_device
    quantiles = model.get_quantiles(X=X_test, q_levels=torch.tensor([0.025, 0.975]).to(output_device))
    return torch.mean(quantiles.T[-1] - quantiles.T[0], 0).to(output_device)


def print_diagnostics(X_test, Y_test, model, rnd=4, rep=None, save_path=None, parallelize=False):

    # nll = round(negative_log_likelihood(X_test, Y_test, model).item(), rnd)
    # print('Negative log likelihood: ', nll)
    ece = round(expected_calibration_error(X_test, Y_test, model).item(), rnd)
    print('Expected calibration error: ', ece)
    avg_interval = round(average_95_interval(X_test, model).item(), rnd)
    print('Average length of 95% interval: ', avg_interval)
    avg_var = round(average_variance(X_test, model).item(), rnd)
    print('Average variance: ', avg_var)
    chk_score = round(check_score(X_test, Y_test, model).item(), rnd)
    print('Check score: ', chk_score)


    # Load past data:
    if parallelize:
        save_path = save_path[:-4] + f'_{rep}' + '.pkl'
    if (rep>0) and not parallelize:
        with open(save_path, 'rb') as f:  # Python 3: open(..., 'rb')
            ECE, INT, VAR, CHK = pickle.load(f)
    else:
        ECE = []
        INT = []
        VAR = []
        CHK = []
    ECE.append(ece), INT.append(avg_interval), VAR.append(avg_var), CHK.append(chk_score)
    # Saving the objects:
    with open(save_path, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([ECE, INT, VAR, CHK], f)

def run_diagnostics(X_test, Y_test, model, rnd=4, display=True):

    # nll = round(negative_log_likelihood(X_test, Y_test, model).item(), rnd)
    # print('Negative log likelihood: ', nll)
    ece = round(expected_calibration_error(X_test, Y_test, model).item(), rnd)
    
    avg_interval = round(average_95_interval(X_test, model).item(), rnd)
    avg_var = round(average_variance(X_test, model).item(), rnd)
    chk_score = round(check_score(X_test, Y_test, model).item(), rnd)
    if display:
        print('Expected calibration error: ', ece)
        print('Average length of 95% interval: ', avg_interval)
        print('Average variance: ', avg_var)
        print('Check score: ', chk_score)

    return np.array([ece, avg_interval, avg_var, chk_score])

def naive_ece(model, X, Y):
    ps = torch.linspace(0.01,0.99,99).to(model.output_device)
    preds = model.get_quantiles(X, ps)
    return torch.mean(torch.abs(torch.tensor([torch.mean((Y.flatten() <= preds[:,i]).float())-ps[i] for i in range(len(ps))]))).item()

def naive_sharpness(model, X, Y_range):
    preds = model.get_quantiles(X, torch.tensor([0.025,0.975]))
    return torch.mean(preds[:,1] - preds[:,0]).detach().cpu().item()/Y_range

def plot_calibration_curve(
    exp_proportions,
    obs_proportions,
    title=None,
    curve_label=None,
    make_plots=False,
):
    # Set figure defaults
    # if make_plots:
    #     width = 5
    #     fontsize = 12
    #     rc = {
    #         "figure.figsize": (width, width),
    #         "font.size": fontsize,
    #         "axes.labelsize": fontsize,
    #         "axes.titlesize": fontsize,
    #         "xtick.labelsize": fontsize,
    #         "ytick.labelsize": fontsize,
    #         "legend.fontsize": fontsize,
    #     }
    #     sns.set(rc=rc)
    #     sns.set_style("ticks")

    #     # Set label
    #     if curve_label is None:
    #         curve_label = "Predictor"

    #     # Plot
    #     plt.figure()
    #     if title is not None:
    #         plt.title(title)
    #     plt.plot([0, 1], [0, 1], "--", label="Ideal")
    #     plt.plot(exp_proportions, obs_proportions, label=curve_label)
    #     plt.fill_between(
    #         exp_proportions, exp_proportions, obs_proportions, alpha=0.2
    #     )
    #     plt.xlabel("Expected proportion in interval")
    #     plt.ylabel("Observed proportion in interval")
    #     plt.axis("square")
    #     buff = 0.01
    #     plt.xlim([0 - buff, 1 + buff])
    #     plt.ylim([0 - buff, 1 + buff])

    #     # Compute miscalibration area
    #     polygon_points = []
    #     for point in zip(exp_proportions, obs_proportions):
    #         polygon_points.append(point)
    #     for point in zip(reversed(exp_proportions), reversed(exp_proportions)):
    #         polygon_points.append(point)
    #     polygon_points.append((exp_proportions[0], obs_proportions[0]))
    #     polygon = Polygon(polygon_points)
    #     x, y = polygon.exterior.xy  # original data
    #     ls = LineString(np.c_[x, y])  # closed, non-simple
    #     lr = LineString(ls.coords[:] + ls.coords[0:1])
    #     mls = unary_union(lr)
    #     polygon_area_list = [poly.area for poly in polygonize(mls)]
    #     miscalibration_area = np.asarray(polygon_area_list).sum()

    #     # Annotate plot with the miscalibration area
    #     plt.text(
    #         x=0.95,
    #         y=0.05,
    #         s="Miscalibration area = %.2f" % miscalibration_area,
    #         verticalalignment="bottom",
    #         horizontalalignment="right",
    #         fontsize=fontsize,
    #     )
    #     plt.show()
    # else:
        # not making plots, just computing ECE
    miscalibration_area = torch.mean(
        torch.abs(exp_proportions - obs_proportions)
    ).item()

    return miscalibration_area


def get_individual_calibration_score(
    exp_proportions,
    obs_proportions
):

    return torch.abs(exp_proportions - obs_proportions)


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

import tqdm
from argparse import Namespace

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
    display=True,
):

    # obs_props, quantile_preds, quantile_preds_mat = \
    #     ens_get_props(model, x, y, exp_props=exp_props, recal_model=recal_model,
    #                   recal_type=recal_type)

    num_pts = x.shape[0]
    ##y = y.detach().cpu().reshape(num_pts, -1)
    y = y.reshape(num_pts, -1)

    quantile_preds = model.get_quantiles(
        x,
        exp_props,
    )  # of shape (num_pts, num_q)
    obs_props = torch.mean((quantile_preds >= y).float(), dim=0).flatten()

    assert exp_props.shape == obs_props.shape
    
    pinball_loss = []
    model_obs = []
    model_y = []
    #pred = model(x)
    #num_quantiles = pred.shape[1]
    mask = (y >= quantile_preds)
    delta = (y - quantile_preds)
    #for i in range(num_quantiles):
    for i, quantile in enumerate(exp_props):
        #quantile = i/(num_quantiles-1)
        q_mask = mask[:,i]
        q_delta = delta[:,i]
        # if i == 0:
        #     print(quantile)
        #     print(torch.mean((~q_mask).float()).item(), torch.mean((~q_mask)*(-q_delta)).item())
        # if i == num_quantiles-1:
        #     print(quantile)
        #     print(torch.mean(q_mask.float()).item(), torch.mean(q_mask*q_delta).item())
        pinball_loss.append(torch.mean(q_mask*q_delta*quantile + (~q_mask)*(-q_delta)*(1-quantile)).item())#If q=0, how many pred<=y?
        model_obs.append(torch.mean((y <= quantile_preds[:,i].reshape(y.shape)).float()).item())
        model_y.append(torch.mean(quantile_preds[:,i]).item())
    
    #print("------------------------------------------------------------------------")    
    # print(y.tolist())
    # print(pred.tolist())
    # print(quantile_preds.tolist())
    # print(model_obs)
    # print(obs_props.tolist())
    # print(model.get_quantiles(x,[0,0.2,0.4,0.6,0.8,1]).tolist())

    idx_01 = get_q_idx(exp_props, 0.01)
    idx_99 = get_q_idx(exp_props, 0.99)
    individual_cali = exp_props[idx_01 : idx_99 + 1] - obs_props[idx_01 : idx_99 + 1]
    cali_score = torch.abs(individual_cali).mean().item()
    # cali_score = plot_calibration_curve(
    #     exp_props[idx_01 : idx_99 + 1],
    #     obs_props[idx_01 : idx_99 + 1],
    #     make_plots=make_plots,
    # )

    order = torch.argsort(y.flatten())
    q_025 = quantile_preds[:, get_q_idx(exp_props, 0.025)][order]
    q_975 = quantile_preds[:, get_q_idx(exp_props, 0.975)][order]
    sharp_score = torch.mean(q_975 - q_025).item() / y_range

    # if make_plots:
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(torch.arange(y.size(0)), q_025)
    #     plt.plot(torch.arange(y.size(0)), q_975)
    #     plt.scatter(torch.arange(y.size(0)), y[order], c="r")
    #     plt.title("Mean Width: {:.3f}".format(sharp_score))
    #     plt.show()

    g_cali_scores = []
    if test_group_cal:
        ratio_arr = np.linspace(0.01, 1.0, 10)
        if display:
            print(
                "Spanning group size from {} to {} in {} increments".format(
                    np.min(ratio_arr), np.max(ratio_arr), len(ratio_arr)
                )
            )
        for r in (tqdm.tqdm(ratio_arr) if display else ratio_arr):
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
    # args = Namespace(
    #     device=torch.device("cpu"), q_list=torch.linspace(0.01, 0.99, 99)
    # )
    # curr_crps = crps_score(model, x, y, args)
    # curr_nll = bag_nll(model, x, y, args)
    # curr_check = check_loss(model, x, y, args)
    # curr_int = interval_score(model, x, y, args)
    # int_exp_props, int_obs_props, int_cdf_preds = get_obs_props(
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

    return (
        cali_score,
        sharp_score,
        obs_props,
        quantile_preds,
        g_cali_scores,
        individual_cali,
        # curr_scoring_rules,
        pinball_loss,
        model_obs,
        model_y
    )