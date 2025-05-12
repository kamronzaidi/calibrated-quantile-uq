import os, sys
import argparse
from argparse import Namespace
from copy import deepcopy
import numpy as np
import pickle as pkl
from typing import Union
import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from data.fetch_data import get_uci_data, get_toy_data, get_fusion_data
from utils.misc_utils import (
    test_uq,
    set_seeds,
    get_q_idx,
    discretize_domain,
    gather_loss_per_q,
)
from recal import iso_recal, iso_recal_ours
from utils.q_model_ens import QModelEns
from utils.Calipso_Model import CalipsoModel
from losses import (
    cali_loss,
    batch_cali_loss,
    qr_loss,
    batch_qr_loss,
    interval_loss,
    batch_interval_loss,
)
from utils.NNKit.models.model import vanilla_nn, bpl_nn, standard_nn_model
from auxiliary.save_results import save_results

def get_loss_fn(loss_name):
    if loss_name == "qr":
        fn = qr_loss
    elif loss_name == "batch_qr":
        fn = batch_qr_loss
    elif loss_name in [
        "cal",
        "scaled_cal",
        "cal_penalty",
        "scaled_cal_penalty",
    ]:
        fn = cali_loss
    elif loss_name in [
        "batch_cal",
        "scaled_batch_cal",
        "batch_cal_penalty",
        "scaled_batch_cal_penalty",
    ]:
        fn = batch_cali_loss
    elif loss_name == "int":
        fn = interval_loss
    elif loss_name == "batch_int":
        fn = batch_interval_loss
    elif "calipso" in loss_name:
        fn = None
    else:
        raise ValueError("loss arg not valid")

    return fn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_ens", type=int, default=1, help="number of members in ensemble"
    )
    parser.add_argument(
        "--boot", type=int, default=0, help="1 to bootstrap samples"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/UCI_Datasets",
        help="parent directory of datasets",
    )
    parser.add_argument(
        "--data", type=str, default="boston", help="dataset to use"
    )
    parser.add_argument(
        "--num_q",
        type=int,
        default=30,
        help="number of quantiles you want to sample each step",
    )
    parser.add_argument("--gpu", type=str, default='0', help="gpu num to use")

    parser.add_argument(
        "--num_ep", type=int, default=600, help="number of epochs"
    )
    parser.add_argument("--nl", type=int, default=2, help="number of layers")
    parser.add_argument("--hs", type=int, default=64, help="hidden size")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
    parser.add_argument("--bs", type=int, default=64, help="batch size")
    parser.add_argument(
        "--wait",
        type=int,
        default=200,
        help="how long to wait for lower validation loss",
    )

    parser.add_argument("--loss", type=str, default='calipso',
                        help="specify type of loss")

    # only for cali losses
    parser.add_argument(
        "--penalty",
        dest="sharp_penalty",#[0-1 in 20 equispaced]
        type=float,
        help="coefficient for sharpness penalty; 0 for none",
    )
    parser.add_argument(
        "--rand_ref",
        type=int,
        help="1 to use rand reference idxs for cali loss",
    )
    parser.add_argument(
        "--sharp_all",
        type=int,
        default=0,
        help="1 to penalize only widths that are over covered",
    )

    # draw a sorted group batch every
    parser.add_argument(
        "--gdp",
        dest="draw_group_every",  #[1, 2, 3, 5, 10, 30, 100]
        type=int,
        help="draw a group batch every # epochs",
    )
    parser.add_argument(
        "--recal", type=int, default=1, help="1 to recalibrate after training"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/",
        help="dir to save results",
    )
    parser.add_argument("--debug", type=int, default=0, help="1 to debug")
    
    parser.add_argument("--skip_dups", type=int, default=1, help="1 to skip over existing files")
    
    parser.add_argument("--big_model", type=int, default=0, help="1 to use bigger model")
    args = parser.parse_args()

    if "penalty" in args.loss:
        assert isinstance(args.sharp_penalty, float)
        assert 0.0 <= args.sharp_penalty <= 1.0

        if args.sharp_all is not None:
            args.sharp_all = bool(args.sharp_all)
    else:
        args.sharp_penalty = None
        args.sharp_all = None

    if args.rand_ref is not None:
        args.rand_ref = bool(args.rand_ref)

    if args.draw_group_every is None:
        args.draw_group_every = args.num_ep

    args.boot = bool(args.boot)
    args.recal = bool(args.recal)
    args.debug = bool(args.debug)

    if args.boot:
        if not args.num_ens > 1:
            raise RuntimeError("num_ens must be above > 1 for bootstrap")

    if args.gpu is str:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    args.device = device

    return args


if __name__ == "__main__":
    # DATA_NAMES = ['wine', 'naval', 'kin8nm', 'energy', 'yacht', 'concrete', 'power', 'boston']

    args = parse_args()

    #print("DEVICE: {}".format(args.device))

    if args.debug:
        import pudb

        pudb.set_trace()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    per_seed_cali = []
    per_seed_sharp = []
    per_seed_gcali = []
    per_seed_crps = []
    per_seed_nll = []
    per_seed_check = []
    per_seed_int = []
    per_seed_int_cali = []
    per_seed_model = []

    print(
        "Drawing group batches every {}, penalty {}".format(
            args.draw_group_every, args.sharp_penalty
        )
    )

    # Save file name
    if "penalty" not in args.loss:
        save_file_name = "{}/{}_loss{}_ens{}_boot{}_gdp{}_seed{}.pkl".format(
            args.save_dir,
            args.data,
            args.loss,
            args.num_ens,
            args.boot,
            args.draw_group_every,
            args.seed,
        )
    else:
        # penalizing sharpness
        if args.sharp_all is not None and args.sharp_all:
            save_file_name = "{}/{}_loss{}_pen{}_sharpall_ens{}_boot{}_gdp{}_seed{}.pkl".format(
                args.save_dir,
                args.data,
                args.loss,
                args.sharp_penalty,
                args.num_ens,
                args.boot,
                args.draw_group_every,
                args.seed,
            )
        elif args.sharp_all is not None and not args.sharp_all:
            save_file_name = "{}/{}_loss{}_pen{}_wideonly_ens{}_boot{}_gdp{}_seed{}.pkl".format(
                args.save_dir,
                args.data,
                args.loss,
                args.sharp_penalty,
                args.num_ens,
                args.boot,
                args.draw_group_every,
                args.seed,
            )
    # if os.path.exists(save_file_name) and args.skip_dups:
    #     print("skipping {}".format(save_file_name))
    #     sys.exit()

    # Set seeds
    set_seeds(args.seed)

    # Fetching data
    data_args = Namespace(
        data_dir=args.data_dir, dataset=args.data, seed=args.seed
    )

    if "uci" in args.data_dir.lower():
        data_out = get_uci_data(args)
    elif "toy" in args.data_dir.lower():
        data_out = get_toy_data(args)

    x_tr, x_va, x_te, y_tr, y_va, y_te, y_al = (
        data_out.x_tr,
        data_out.x_va,
        data_out.x_te,
        data_out.y_tr,
        data_out.y_va,
        data_out.y_te,
        data_out.y_al,
    )
    y_range = (y_al.max() - y_al.min()).item()
    # print("y range: {:.3f}".format(y_range))

    # Making models
    num_tr = x_tr.shape[0]
    dim_x = x_tr.shape[1]
    dim_y = y_tr.shape[1]
    if "calipso" in args.loss:
        model_ens = CalipsoModel(
            input_size=dim_x,
            output_size=dim_y,
            hidden_size=args.hs,
            num_layers=args.nl,
            lr=args.lr,
            wd=args.wd,
            num_ens=args.num_ens,
            device=args.device,
            big_model=args.big_model,
            cali_X=x_tr,
            cali_y=y_tr,
            pretrain="pretrain" in args.loss
        )
    else:
        model_ens = QModelEns(
            input_size=dim_x + 1,
            output_size=dim_y,
            hidden_size=args.hs,
            num_layers=args.nl,
            lr=args.lr,
            wd=args.wd,
            num_ens=args.num_ens,
            device=args.device,
            big_model=args.big_model
        )
    # Data loader
    if not args.boot:
        loader = DataLoader(
            TensorDataset(x_tr, y_tr),
            shuffle=True,
            batch_size=args.bs,
        )
    else:
        rand_idx_list = [
            np.random.choice(num_tr, size=num_tr, replace=True)
            for _ in range(args.num_ens)
        ]
        loader_list = [
            DataLoader(
                TensorDataset(x_tr[idxs], y_tr[idxs]),
                shuffle=True,
                batch_size=args.bs,
            )
            for idxs in rand_idx_list
        ]

    # Loss function
    loss_fn = get_loss_fn(args.loss)
    args.scale = True if "scale" in args.loss else False
    batch_loss = True if "batch" in args.loss else False

    """ train loop """
    tr_loss_list = []
    va_loss_list = []
    te_loss_list = []

    # setting batch groupings
    group_list = discretize_domain(x_tr.numpy(), args.bs)
    curr_group_idx = 0
    desired_eces = torch.linspace(0, 0.2, 20)
    best_sharp_scores = [float('inf') for _ in range(len(desired_eces))]
    best_models = [None for _ in range(len(desired_eces))]


    for ep in tqdm.tqdm(range(args.num_ep)):
        if model_ens.done_training:
            # print("Done training ens at EP {}".format(ep))
            break

        # Take train step
        # list of losses from each batch, for one epoch
        ep_train_loss = []
        if not args.boot:
            if ep % args.draw_group_every == 0:
                # drawing a group batch
                group_idxs = group_list[curr_group_idx]
                curr_group_idx = (curr_group_idx + 1) % dim_x
                for g_idx in group_idxs:
                    xi = x_tr[g_idx.flatten()].to(args.device)
                    yi = y_tr[g_idx.flatten()].to(args.device)
                    q_list = torch.rand(args.num_q)
                    #print(xi.shape)
                    loss = model_ens.loss(
                        loss_fn,
                        xi,
                        yi,
                        q_list,
                        batch_q=batch_loss,
                        take_step=True,
                        args=args,
                    )
                    ep_train_loss.append(loss)
            else:
                # just doing ordinary random batch
                for (xi, yi) in loader:
                    xi, yi = xi.to(args.device), yi.to(args.device)
                    q_list = torch.rand(args.num_q)
                    loss = model_ens.loss(
                        loss_fn,
                        xi,
                        yi,
                        q_list,
                        batch_q=batch_loss,
                        take_step=True,
                        args=args,
                    )
                    ep_train_loss.append(loss)
        else:
            # bootstrapped ensemble of models
            for xi_yi_samp in zip(*loader_list):
                xi_list = [item[0].to(args.device) for item in xi_yi_samp]
                yi_list = [item[1].to(args.device) for item in xi_yi_samp]
                assert len(xi_list) == len(yi_list) == args.num_ens
                q_list = torch.rand(args.num_q)
                loss = model_ens.loss_boot(
                    loss_fn,
                    xi_list,
                    yi_list,
                    q_list,
                    batch_q=batch_loss,
                    take_step=True,
                    args=args,
                )
                ep_train_loss.append(loss)
        ep_tr_loss = np.nanmean(np.stack(ep_train_loss, axis=0), axis=0)
        tr_loss_list.append(ep_tr_loss)
        
        # Validation loss
        x_va, y_va = x_va.to(args.device), y_va.to(args.device)
        va_te_q_list = torch.linspace(0.01, 0.99, 99)
        ep_va_loss = model_ens.update_va_loss(
            loss_fn,
            x_va,
            y_va,
            va_te_q_list,
            batch_q=batch_loss,
            curr_ep=ep,
            num_wait=args.wait,
            args=args,
        )
        va_loss_list.append(ep_va_loss)

        # Test loss
        x_te, y_te = x_te.to(args.device), y_te.to(args.device)
        with torch.no_grad():
            ep_te_loss = model_ens.loss(
                loss_fn,
                x_te,
                y_te,
                va_te_q_list,
                batch_q=batch_loss,
                take_step=False,
                args=args,
            )
        te_loss_list.append(ep_te_loss)

        x_tr, y_tr, x_va, y_va, x_te, y_te = (
            x_tr.cpu(),
            y_tr.cpu(),
            x_va.cpu(),
            y_va.cpu(),
            x_te.cpu(),
            y_te.cpu(),
        )
        model_ens.use_device(torch.device("cpu"))

        # THE NEXT FEW LINES ARE TOO LONG AND TOO COMPLICATED. TODO: Use a better way (that doesn't require putting the model on cpu)!
        if "calipso" in args.loss:
            #Calibrate remaining quantiles
            # tr_q_list = torch.linspace(0.01, 0.99, 99)
            tr_exp_props = torch.linspace(-0.01, 1.01, 103)
            tr_cali_score, tr_sharp_score, tr_obs_props, tr_q_preds, _, _ = test_uq(
                model_ens,
                x_tr,
                y_tr,
                tr_exp_props,
                y_range,
                recal_model=None,
                recal_type=None,
            )
            recal_model_calipso = iso_recal(tr_exp_props, tr_obs_props)
        else:
            recal_model_calipso = None

        # Test UQ on val
        # print("Testing UQ on val")
        va_exp_props = torch.linspace(-2.0, 3.0, 501)
        va_cali_score, va_sharp_score, va_obs_props, va_q_preds, _, _ = test_uq(
        model_ens,
        x_va,
        y_va,
        va_exp_props,
        y_range,
        recal_model=recal_model_calipso,
        recal_type="sklearn",
        )

        for i, ece in enumerate(desired_eces):
            if va_cali_score<ece and va_sharp_score < best_sharp_scores[i]:
                best_sharp_scores[i] = va_sharp_score
                if "calipso" in args.loss:
                    model_ens.alphas = [alpha.detach() for alpha in model_ens.alphas]
                best_models[i] = deepcopy(model_ens)
    
        x_tr, y_tr, x_va, y_va, x_te, y_te = (
            x_tr.to(args.device),
            y_tr.to(args.device),
            x_va.to(args.device),
            y_va.to(args.device),
            x_te.to(args.device),
            y_te.to(args.device),
        )
        model_ens.use_device(args.device)
    

        # Printing some losses
        # if (ep % 200 == 0) or (ep == args.num_ep - 1):
        #     print("EP:{}".format(ep))
        #     print("Train loss {}".format(ep_tr_loss))
        #     print("Val loss {}".format(ep_va_loss))
        #     print("Test loss {}".format(ep_te_loss))

    # Finished training
    save_results(x_tr, y_tr, x_va, y_va, x_te, y_te, best_models, desired_eces, model_ens, args, y_range, tr_loss_list, va_loss_list, te_loss_list, best_sharp_scores, save_file_name)