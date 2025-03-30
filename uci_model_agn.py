import os
import sys
import random
import numpy as np
import pickle as pkl
import argparse
from argparse import Namespace
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.stats import norm, iqr
import seaborn as sns
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
#sys.path.append('libs')
#from libs.NNKit.models.model import vanilla_nn
from utils.NNKit.models.model import vanilla_nn, bpl_nn, standard_nn_model
from recal import iso_recal, iso_recal_ours
from normal_regression import gen_model

from utils.q_model_ens import QModelEns
from utils.misc_utils import test_uq

def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot_calibration_curve(exp_proportions, obs_proportions,
                           title=None, curve_label=None):
    if curve_label is None:
        miscalibration_area = np.mean(np.abs(exp_proportions.flatten() - obs_proportions.flatten()))

    else:
        # Set figure defaults
        width = 5
        fontsize = 12
        rc = {'figure.figsize': (width, width),
            'font.size': fontsize,
            'axes.labelsize': fontsize,
            'axes.titlesize': fontsize,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            'legend.fontsize': fontsize}
        sns.set(rc=rc)
        sns.set_style('ticks')

        # Set label
        if curve_label is None:
            curve_label = 'Predictor'
        # Plot
        plt.figure()
        if title is not None:
            plt.title(title)
        plt.plot([0, 1], [0, 1], '--', label='Ideal')
        plt.plot(exp_proportions, obs_proportions, label=curve_label)
        plt.fill_between(exp_proportions, exp_proportions, obs_proportions,
                       alpha=0.2)
        plt.xlabel('Expected proportion in interval')
        plt.ylabel('Observed proportion in interval')
        plt.axis('square')
        buff = 0.01
        plt.xlim([0-buff, 1+buff])
        plt.ylim([0-buff, 1+buff])

        # Compute miscalibration area
        polygon_points = []
        for point in zip(exp_proportions, obs_proportions):
            polygon_points.append(point)
        for point in zip(reversed(exp_proportions), reversed(exp_proportions)):
            polygon_points.append(point)
        polygon_points.append((exp_proportions[0], obs_proportions[0]))
        polygon = Polygon(polygon_points)
        x, y = polygon.exterior.xy # original data
        ls = LineString(np.c_[x, y]) # closed, non-simple
        lr = LineString(ls.coords[:] + ls.coords[0:1])
        mls = unary_union(lr)
        polygon_area_list =[poly.area for poly in polygonize(mls)]
        miscalibration_area = np.asarray(polygon_area_list).sum()

        # Annotate plot with the miscalibration area
        plt.text(x=0.95, y=0.05,
               s='Miscalibration area = %.2f' % miscalibration_area,
               verticalalignment='bottom',
               horizontalalignment='right',
               fontsize=fontsize)
        plt.show()
    return miscalibration_area


def ecdf(points, weights):
    """
    creates an (weighted) empirical cdf with the given points
    Args:
        points: array of points to construct cdf
                ...these are the residuals
                (num_points, or num_points x 1)
        weights: array of weights to assign to each point
    """
    points, weights = points.flatten(), weights.flatten()
    order = np.argsort(points, axis=0).flatten()
    if weights is None:
        weights = np.ones_like(points) / (len(points))
    else:
        weights = weights / (np.sum(weights))
    probs = []
    cum_prob = 0
    for w in weights[order]:
        cum_prob += w
        probs.append(cum_prob)
    return np.array(probs), order


def get_dist_matrix(x_arr, type, kernel_length_scales, num_in_bin):
    """
    Given an array of X points,
    we calculate the distance of each point to all other points
    and return this distance matrix

    Args:
        x_arr: array in X space (num_points, dim)
        type: type of distance to take (one of 'euclidean' or 'kernel')
        kernel_length_scale: used for kernel distance
        num_in_bin: used to report the average distance for this many
                    points to be collected for each point
    """

    num_pts = x_arr.shape[0]
    interim_mat = None
    dists_list = []

    #print('Making dist matrix')
    for pt_idx in tqdm.tqdm(range(num_pts)):
        curr_pt = x_arr[pt_idx]

        if type == 'euclidean':
            diffs = x_arr - curr_pt
            dists = np.linalg.norm(diffs, axis=1).flatten()
            assert dists.size == num_pts
            dists_list.append(dists)

        elif type == 'kernel':
            if kernel_length_scales is None:
                raise ValueError('must specify kernel_length_scale when using kernel')
            se_kernel = RBF(length_scale=kernel_length_scales)
            kernel_weights = se_kernel(curr_pt, x_arr).flatten()
            assert kernel_weights.size == num_pts
            dists_list.append(kernel_weights)

        else:
            raise ValueError("type must be one of 'euclidean' or 'kernel'")
    dists_per_point = np.stack(dists_list, axis=0)
    return dists_per_point


def get_resid_for_target_p(probs, resid_order, resids, target_p,
                           min_resid, max_resid):
    probs, resid_order, resids = \
        probs.flatten(), resid_order.flatten(), resids.flatten()

    if target_p < probs.min():
        prev_p, next_p = 0, probs.min()
        prev_resid = 2*resids.min() - np.sort(resids)[1]
        next_resid = resids.min()
    elif target_p > probs.max():
        raise ValueError('target_p cannot be bigger than 1.0')
    else:
        for i, p in enumerate(probs):
            if p > target_p:
                prev_idx, next_idx = i - 1, i
                break
        prev_p, next_p = probs[prev_idx], probs[next_idx]
        prev_resid, next_resid = resids[resid_order][prev_idx], \
                                 resids[resid_order][next_idx]

    target_prob_quantile = (target_p - prev_p) / (next_p - prev_p)
    target_resid = prev_resid + (next_resid - prev_resid) * target_prob_quantile

    return target_resid


def make_cdf_dataset(train_domain, train_obs, mean_pred,
                     construct_method, hps, tr_beg_idx, tr_end_idx,
                     alpha=None):
    """
    Returns the quantile model dataset

    Args:
        train_domain: array of x points
        train_obs: array of y points
        mean_pred: mean prediction given train_domain
        construct_method: in ['bin', 'bin_wecdf', 'kernel_wecdf',
                              'weighted_kdecdf']
        hps: Namespace with hyperparams
        tr_beg_idx: begin at this index of train_domain
        tr_end_idx: end at this index of train_domain
        alpha: if not None, create dataset only for the PI (alpha/2, 1-(alpha/2))
    """

    #print('kde bandwidth is {}'.format(hps.resid_bandwidth))

    if construct_method not in ['bin', 'bin_wecdf', 'kernel_wecdf',
                                'weighted_kdecdf']:
        raise ValueError('construct_method must be one of bin, wecdf, wkcdf')

    num_train = train_domain.shape[0]
    all_resids = train_obs - mean_pred
    min_resid, max_resid = all_resids.min(), all_resids.max()

    e_train_domain = []
    e_train_obs = []

    if alpha is not None:
        p_low = alpha / 2.0
        p_high = 1 - (alpha / 2.0)
        etr_x_low_list = []
        etr_y_low_list = []
        etr_x_high_list = []
        etr_y_high_list = []

    #print('Constructing icdf dataset...')
    for pt_idx in range(tr_beg_idx, tr_end_idx):
        pt = train_domain[pt_idx].reshape(1, -1)
        ######## Binning on L2 distance
        if construct_method in ['bin', 'bin_wecdf']:
            x_diffs = train_domain - pt
            x_dists = np.linalg.norm(x_diffs, axis=1).flatten()

            neighbor_idxs = (x_dists <= hps.dist_thresh)
            resids = all_resids[neighbor_idxs]

            if construct_method == 'bin':
                neighbor_weights = np.ones_like(resids)

            elif construct_method == 'bin_wecdf':
                neighbor_dists = x_dists[neighbor_idxs]
                neighbor_weights = np.max(neighbor_dists) - neighbor_dists + 1e-5
            # below, probs will always be increasing
            probs, resid_order = ecdf(resids, neighbor_weights)

            if probs.size < 2:
                #print('skipping {}'.format(pt_idx))
                continue

            if alpha is not None:
                p_low_resid = get_resid_for_target_p(probs, resid_order, resids,
                                                     p_low, min_resid, max_resid)
                p_high_resid = get_resid_for_target_p(probs, resid_order, resids,
                                                      p_high, min_resid, max_resid)

        ######## SE kernel
        elif construct_method == 'kernel_wecdf':
            """
            setting weights for each point in train_domain with a kernel
            hps must have:
                1) kernel_length_scales
                2) prune_kernel_weights: whether to filter X points based on
                                         dist induced by kernel
                3) kernel_weight_thresh: threshold to filter X points based on
                                         kernel dist
            """
            se_kernel = RBF(length_scale=hps.kernel_length_scales)
            kernel_weights = se_kernel(pt, train_domain).flatten()
            if hps.prune_kernel_weights:
                # idx of points for which weights by kernel is above thresh
                above_kernel_thresh = (kernel_weights >= hps.kernel_weight_thresh)
                resids = all_resids[above_kernel_thresh]
                kernel_weights = kernel_weights[above_kernel_thresh]
                kernel_weights = kernel_weights / np.sum(kernel_weights)
            else:
                resids = all_resids
            probs, resid_order = ecdf(resids, kernel_weights)

            if probs.size < 2:
                #print('skipping {}'.format(pt_idx))
                continue

            if alpha is not None:
                p_low_resid = get_resid_for_target_p(probs, resid_order, resids,
                                                     p_low, min_resid, max_resid)
                p_high_resid = get_resid_for_target_p(probs, resid_order, resids,
                                                      p_high, min_resid, max_resid)

        elif construct_method == 'weighted_kdecdf':
            """
            construct a cdf in the y-space with KDE (can ONLY use gaussian
            kernel densities because of its closed form cdf)
            hps must have:
                1) kernel_length_scales
                2) prune_kernel_weights: whether to filter X points based on
                                        dist induced by kernel
                3) kernel_weight_thresh: threshold to filter X points based on
                                        kernel dist
                4) num_cdf_query: number of residual points to query
                                  constructed cdf
                5) resid_bandwidth: bandwidth of gaussian densities for each
                                    y pt to construct KDE
            """
            se_kernel = RBF(length_scale=hps.kernel_length_scales)
            kernel_weights = se_kernel(pt, train_domain).flatten()
            # making residual points to query constructed cdf
            num_cdf_query = hps.num_cdf_query
            # make range of query 2 percent wider
            range_frac = (max_resid - min_resid) / 100.
            cdf_query_points = np.linspace(min_resid - range_frac,
                                           max_resid + range_frac,
                                           num_cdf_query)
            if hps.prune_kernel_weights:
                # idx of points for which weights by kernel is above thresh
                above_kernel_thresh = (kernel_weights >= hps.kernel_weight_thresh)
                resids = all_resids[above_kernel_thresh]
                kernel_weights = kernel_weights[above_kernel_thresh]
                kernel_weights = kernel_weights / np.sum(kernel_weights)
            else:
                resids = all_resids

            """ make kde bandwidth """
            num_resid = resids.shape[0]
            if num_resid < 2:
                #print('skipping {}'.format(pt_idx))
                continue

            resid_std = np.std(resids.flatten())
            resid_iqr = iqr(resids.flatten())

            if hps.resid_bandwidth is None:
                hps.resid_bandwidth = \
                0.9 * np.min([resid_std, resid_iqr]) * (num_resid ** (-1/5))

            cdf_vals = norm.cdf(x=cdf_query_points.reshape(num_cdf_query, 1),
                                loc=resids.flatten(),
                                scale=hps.resid_bandwidth *
                                      np.ones_like(resids.flatten()))
            cdf_vals = kernel_weights * cdf_vals
            cdf_vals = np.mean(cdf_vals, axis=1)
            probs = ((cdf_vals - cdf_vals.min()) /
                     (cdf_vals.max() - cdf_vals.min()))
            resids = cdf_query_points
            resid_order = np.arange(num_cdf_query)

            if alpha is not None:
                p_low_resid = get_resid_for_target_p(probs, resid_order, resids,
                                                     p_low, min_resid, max_resid)
                p_high_resid = get_resid_for_target_p(probs, resid_order, resids,
                                                      p_high, min_resid, max_resid)

        if probs.size == 0:
            continue

        """ construct ecdf dataset """
        if alpha is None:
            # probs is a flat array
            pt_filled = np.repeat(pt, probs.size, axis=0)
            curr_e_train_x = np.concatenate([pt_filled, probs.reshape(-1, 1)],
                                            axis=1)
            e_train_domain.append(curr_e_train_x)
            e_train_obs.append(resids[resid_order])
        else:
            curr_etr_x_low = np.concatenate([pt, np.array(p_low).reshape(1,1)], axis=1)
            curr_etr_x_high = np.concatenate([pt, np.array(p_high).reshape(1,1)], axis=1)

            etr_x_low_list.append(curr_etr_x_low)
            etr_y_low_list.append(p_low_resid)

            etr_x_high_list.append(curr_etr_x_high)
            etr_y_high_list.append(p_high_resid)

    if alpha is None:
        e_train_domain = np.concatenate(e_train_domain, axis=0)
        e_train_obs = np.concatenate(e_train_obs, axis=0)
        return e_train_domain, e_train_obs
    else:
        etr_x_low = np.concatenate(etr_x_low_list, axis=0)
        etr_y_low = np.vstack(etr_y_low_list)

        etr_x_high = np.concatenate(etr_x_high_list, axis=0)
        etr_y_high = np.vstack(etr_y_high_list)

        return etr_x_low, etr_y_low, etr_x_high, etr_y_high


def create_data_split(x_al, y_al, seed):
    x_tr, x_te, y_tr, y_te = train_test_split(
        x_al, y_al, test_size=0.1, random_state=seed)
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_tr, y_tr, test_size=0.2, random_state=seed)

    s_tr_x = StandardScaler().fit(x_tr)
    s_tr_y = StandardScaler().fit(y_tr)


    x_tr = s_tr_x.transform(x_tr)
    x_tr = torch.Tensor(x_tr)
    x_val = s_tr_x.transform(x_val)
    x_val = torch.Tensor(x_val)
    x_te = s_tr_x.transform(x_te)
    x_te = torch.Tensor(x_te)

    y_tr = s_tr_y.transform(y_tr)
    y_tr = torch.Tensor(y_tr)
    y_val = s_tr_y.transform(y_val)
    y_val = torch.Tensor(y_val)
    y_te = s_tr_y.transform(y_te)
    y_te = torch.Tensor(y_te)
    y_al = torch.Tensor(s_tr_y.transform(y_al))

    return x_tr, y_tr, x_val, y_val, x_te, y_te, y_al


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--n_hidden_layers', type=int, default=2)
    parser.add_argument('--n_hidden_units', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--overfit_length', type=int, default=200)
    parser.add_argument('--dist_type', type=str, default='kernel')
    parser.add_argument('--num_in_bin', type=int, default=40)
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--recal', type=int, default=0)
    parser.add_argument('--big_model', type=int, default=0)
    args = parser.parse_args()
    if bool(args.debug):
        import pudb; pudb.set_trace()

    if args.gpu == '':
        use_gpu=False
        device = torch.device('cpu')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    if args.dataset == '':
        datasets = ['yacht','boston','energy','concrete','wine','power','kin8nm','naval','protein']
    else:
        datasets = [args.dataset]
    
    if args.seed == -1:
        SEEDS = [0, 1, 2, 3, 4]
    else:
        SEEDS = [args.seed]

    for dataset in datasets:
        print("Dataset:",dataset)
        per_seed_cali = []
        per_seed_sharp = []
        per_seed_gcali = []
        per_seed_crps = []
        per_seed_nll = []
        per_seed_check = []
        per_seed_int = []
        per_seed_int_cali = []
        per_seed_model = []
        per_seed_recal_cali = []
        per_seed_recal_sharp = []
        
        for s in tqdm.tqdm(SEEDS):
            print(f"Seed: {s}")
            """ set seed """
            seed = s
            reset_seeds(seed)

            """ load data """
            data = np.loadtxt('data/UCI_Datasets/{}.txt'.format(dataset))
            x_al = data[:, :-1]
            y_al = data[:, -1].reshape(-1, 1)

            x_tr, y_tr, x_va, y_va, x_te, y_te, y_al = create_data_split(x_al, y_al, seed)
            y_range = (y_al.max() - y_al.min()).item()

            if use_gpu:
                loader_tr = DataLoader(TensorDataset(x_tr, y_tr),
                                    shuffle=False,
                                    batch_size=args.bs,
                                    pin_memory=True)
            else:
                loader_tr = DataLoader(TensorDataset(x_tr, y_tr),
                                    shuffle=True,
                                    batch_size=args.bs)
            x_tr = x_tr.to(device)
            y_tr = y_tr.to(device)
            x_va = x_va.to(device)
            x_va2 = x_va.to(device)
            y_va = y_va.to(device)
            y_va2 = y_va.to(device)
            x_te = x_te.to(device)
            y_te = y_te.to(device)

            num_tr = y_tr.shape[0]
            num_va = y_va.shape[0]
            num_te = y_te.shape[0]

            """ load mean model """
            #Changed for ICML Rebuttal
            if args.big_model:
                f_path = f'big_mean_models/{dataset}_{seed}_mean.pkl'
            else:
                f_path = f'mean_models/{dataset}_{seed}_mean.pkl'
                
            if not os.path.exists(f_path):
                if args.big_model:
                    gen_model(dataset, seed, f_path, device, base_model = standard_nn_model)
                else:
                    gen_model(dataset, seed, f_path, device) #Technically wrong, needs to use consistent model, but very little diff
            with open(f_path, 'rb') as pf:
                mean_model = pkl.load(pf)
                mean_model.to(device)

            """ make mean preds """
            with torch.no_grad():
                pred_tr = mean_model(x_tr).cpu().numpy()
                pred_va = mean_model(x_va).cpu().numpy()
                pred_te = mean_model(x_te).cpu().numpy()

            """ merge train and val preds """
            pred_tr_va = np.concatenate([pred_tr, pred_va], axis=0)
            x_tr_va = np.concatenate([x_tr.cpu().numpy(), x_va.cpu().numpy()], axis=0)
            y_tr_va = np.concatenate([y_tr.cpu().numpy(), y_va.cpu().numpy()], axis=0)

            """ get dist matrix """
            dist_type = args.dist_type
            kls = np.ones(x_tr.shape[1])
            dist_mat = get_dist_matrix(x_arr=x_tr_va, type=dist_type,
                                    kernel_length_scales=kls, num_in_bin=20)

            """ set dist threshold """
            num_in_bin = args.num_in_bin
            if dist_type == 'euclidean':
                dist_thresh = np.mean(np.sort(dist_mat, axis=1)[:,num_in_bin])
            elif dist_type == 'kernel':
                dist_thresh = np.mean(np.sort(dist_mat, axis=1)[:,-1*(num_in_bin+1)])
            #print('{} distance set to {:.4f}'.format(dist_type, dist_thresh))

            """ create hps Namespace """
            hps = Namespace()
            hps.dist_thresh = dist_thresh
            hps.kernel_weight_thresh = dist_thresh
            hps.prune_kernel_weights = True
            hps.kernel_length_scales = kls
            # if weighted_kdecdf
            hps.num_cdf_query = 40
            hps.resid_bandwidth = 0.1

            """ create quantile model dataset """
            e_train_domain, e_train_obs = \
                make_cdf_dataset(train_domain=x_tr_va, train_obs=y_tr_va,
                                mean_pred=pred_tr_va,
                                construct_method='kernel_wecdf', hps=hps,
                                tr_beg_idx=0, tr_end_idx=(num_tr+num_va))

            #import pudb; pudb.set_trace()
            cdf_x_tr, cdf_x_va, cdf_y_tr, cdf_y_va = train_test_split(
                e_train_domain, e_train_obs, test_size=0.2, random_state=seed)

            cdf_x_tr_tensor = torch.Tensor(cdf_x_tr)
            cdf_y_tr_tensor = torch.Tensor(cdf_y_tr)
            cdf_x_va_tensor = torch.Tensor(cdf_x_va)
            cdf_y_va_tensor = torch.Tensor(cdf_y_va)

            loader_cdf = DataLoader(TensorDataset(cdf_x_tr_tensor, cdf_y_tr_tensor),
                                    shuffle=True,
                                    batch_size=args.bs)

            """ training quantile model """
            lr = args.lr
            wd = args.wd
            num_epoch = args.n_epochs
            torch_mse = torch.nn.MSELoss()

            # cdf_model = vanilla_nn(input_size=cdf_x_tr_tensor.size(1),
            #                        output_size=1,
            #                        bias=1,
            #                        hidden_size=args.n_hidden_units,
            #                        num_layers=args.n_hidden_layers,
            #                        use_bn=False,
            #                        actv_type='relu',
            #                        softmax=False)
            # cdf_model = cdf_model.to(device)
            # cdf_optimizer = torch.optim.Adam(cdf_model.parameters(), lr=lr, weight_decay=wd)
            cdf_model = QModelEns(input_size=cdf_x_tr.shape[1],
                                output_size=1, hidden_size=args.n_hidden_units,
                                num_layers=args.n_hidden_layers,
                                lr=args.lr, wd=args.wd,
                                num_ens=1, device=device)

            def mse_loss_fn(model, y, x, q_list, device, args):
                pred = model(x)
                loss = torch_mse(y, pred)
                return loss


            """ train loop """
            #print('Training Quantile Model...')
            cdf_best_model = None
            va_loss = []
            for epoch in tqdm.tqdm(range(num_epoch)):

                if cdf_model.done_training:
                    #print('Done training ens at EP {}'.format(epoch))
                    break

                for (xi, yi) in loader_cdf:
                    xi, yi = xi.to(device), yi.to(device)
                    #####
                    loss = cdf_model.loss(mse_loss_fn, xi, yi, q_list=None, batch_q=True, take_step=True, args=args)
                    #####


                    # cdf_optimizer.zero_grad()
                    # pred = cdf_model(xi)
                    # loss = torch_mse(yi, pred)
                    # loss.backward()
                    # cdf_optimizer.step()

                x_va, y_va = cdf_x_va_tensor.to(device), cdf_y_va_tensor.to(device)
                ep_va_loss = cdf_model.update_va_loss(mse_loss_fn, x_va, y_va, q_list=None, batch_q=True, curr_ep=epoch, num_wait=args.overfit_length, args=args)

                # """ get validation loss"""
                # with torch.no_grad():
                #     va_pred = cdf_model.model[0](cdf_x_va_tensor.to(device))
                #     mse_va = torch_mse(cdf_y_va_tensor.to(device), va_pred)
                #     va_loss.append(round(mse_va.item(),6))

                # if (epoch == 0) or (mse_va < np.min(va_loss)):
                #     cdf_best_model = deepcopy(cdf_model)

                # if epoch - np.argmin(va_loss) > args.overfit_length:
                #     print('{}: stopping training'.format(epoch))
                #     break

            x_tr, y_tr, x_va, y_va, x_te, y_te = \
                x_tr.cpu(), y_tr.cpu(), x_va.cpu(), y_va.cpu(), x_te.cpu(), y_te.cpu()
            cdf_model.use_device(torch.device('cpu'))

            y_va_centered = y_va2.cpu() - torch.from_numpy(pred_va)
            y_te_centered = y_te - torch.from_numpy(pred_te)
            """ test calibration """
            #print('Testing UQ on test')
            te_exp_props = torch.linspace(0.01, 0.99, 99)
            te_cali_score, te_sharp_score, te_obs_props, te_q_preds, \
            te_g_cali_scores, te_scoring_rules = \
                test_uq(cdf_model, x_te, y_te_centered, te_exp_props, y_range,
                        recal_model=None, recal_type=None, test_group_cal=True)

            per_seed_cali.append(te_cali_score)
            per_seed_sharp.append(te_sharp_score)
            per_seed_gcali.append(te_g_cali_scores)
            per_seed_crps.append(te_scoring_rules['crps'])
            per_seed_nll.append(te_scoring_rules['nll'])
            per_seed_check.append(te_scoring_rules['check'])
            per_seed_int.append(te_scoring_rules['int'])
            per_seed_int_cali.append(te_scoring_rules['int_cali'])
            cdf_model.use_device(torch.device('cpu'))
            per_seed_model.append(cdf_model)
            
            #print('\n')
            #print('-' * 80)
            #print(args.dataset)
            print('Test Cali: {:.3f}, Sharp: {:.3f}'.format(
                te_cali_score, te_sharp_score))
            #print(te_g_cali_scores[:5])
            #print(te_g_cali_scores[5:])
            #print(te_scoring_rules)
            #print('-'*80)
                        
            if args.recal:
                va_exp_props = torch.linspace(-2.0, 3.0, 501)
                va_cali_score, va_sharp_score, va_obs_props, va_q_preds, _, _ = test_uq(cdf_model, x_va2.cpu(), y_va_centered, va_exp_props, y_range,
                        recal_model=None, recal_type=None)
                recal_model = iso_recal(va_exp_props, va_obs_props)
                #recal_model = iso_recal_ours(cdf_model, x_va2.cpu(), y_va_centered)
                recal_exp_props = torch.linspace(0.01, 0.99, 99)
                (
                    recal_te_cali_score,
                    recal_te_sharp_score,
                    recal_te_obs_props,
                    recal_te_q_preds,
                    recal_te_g_cali_scores,
                    recal_te_scoring_rules
                ) = test_uq(cdf_model, x_te, y_te_centered, te_exp_props, y_range,
                                recal_model=recal_model, recal_type="sklearn", test_group_cal=True,)
                print('Recal Test Cali: {:.3f}, Sharp: {:.3f}'.format(
                recal_te_cali_score, recal_te_sharp_score))
                per_seed_recal_cali.append(recal_te_cali_score)
                per_seed_recal_sharp.append(recal_te_sharp_score)
            



        print('Cali: {}'.format(np.mean(per_seed_cali)))
        print('Sharp: {}'.format(np.mean(per_seed_sharp)))
        print('Recal Cali: {}'.format(np.mean(per_seed_recal_cali)))
        print('Recal Sharp: {}'.format(np.mean(per_seed_recal_sharp)))
        #print('NLL: {}'.format(np.mean(per_seed_nll)))
        #print('CRPS: {}'.format(np.mean(per_seed_crps)))
        #print('Check: {}'.format(np.mean(per_seed_check)))
        #print('Int: {}'.format(np.mean(per_seed_int)))
        #print('Int-Cali: {}'.format(np.mean(per_seed_int_cali)))
        mean_gcali = np.mean(np.stack(per_seed_gcali, axis=0), axis=0)
        #print(mean_gcali[:5])
        #print(mean_gcali[5:])

        save_package = {
            'args': args,
            'per_seed_cali': per_seed_cali,
            'per_seed_sharp': per_seed_sharp,
            'per_seed_gcali': per_seed_gcali,
            'per_seed_crps': per_seed_crps,
            'per_seed_nll': per_seed_nll,
            'per_seed_check': per_seed_check,
            'per_seed_int': per_seed_int,
            'per_seed_int_cali': per_seed_int_cali,
            'per_seed_model': per_seed_model,
            'per_seed_recal_cali': per_seed_recal_cali,
            'per_seed_recal_sharp': per_seed_recal_sharp
        }
        label = f'mauq_recal_{args.recal}_bigmodel_{args.big_model}'
        save_name = '{}_{}_bin{}.pkl'.format(
            dataset, label, dist_thresh)

        #import pdb; pdb.set_trace()
        with open(save_name, 'wb') as pf:
            pkl.dump(save_package, pf)

