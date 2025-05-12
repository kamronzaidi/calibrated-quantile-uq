import pandas as pd
from scipy.io import arff
import torch
import pickle as pkl
from utils.misc_utils import test_uq, get_q_idx
from recal import iso_recal

from sklearn.preprocessing import StandardScaler

def save_results(x_tr, y_tr, x_va, y_va, x_te, y_te, best_models, desired_eces, model_ens, args, y_range, tr_loss_list, va_loss_list, te_loss_list, best_sharp_scores, save_file_name):
        # Move everything to cpu
    x_tr, y_tr, x_va, y_va, x_te, y_te = (
        x_tr.cpu(),
        y_tr.cpu(),
        x_va.cpu(),
        y_va.cpu(),
        x_te.cpu(),
        y_te.cpu(),
    )
    model_ens.use_device(torch.device("cpu"))
    desired_eces = [desired_eces[i] for i in range(len(best_models)) if best_models[i] is not None]
    best_models = [model for model in best_models if model is not None]
    for model in best_models:
        model.use_device(torch.device("cpu"))

    va_cali_score = [0.0 for _ in range(len(best_models))]
    va_sharp_score = [0.0 for _ in range(len(best_models))]
    va_exp_props = [0.0 for _ in range(len(best_models))]
    va_obs_props = [0.0 for _ in range(len(best_models))]
    va_q_preds = [0.0 for _ in range(len(best_models))]
    te_cali_score = [0.0 for _ in range(len(best_models))]
    te_sharp_score = [0.0 for _ in range(len(best_models))]
    te_exp_props = [0.0 for _ in range(len(best_models))]
    te_obs_props = [0.0 for _ in range(len(best_models))]
    te_q_preds = [0.0 for _ in range(len(best_models))]
    te_g_cali_scores = [0.0 for _ in range(len(best_models))]
    te_scoring_rules = [0.0 for _ in range(len(best_models))]
    recal_va_cali_score = [0.0 for _ in range(len(best_models))]
    recal_va_sharp_score = [0.0 for _ in range(len(best_models))]
    recal_va_obs_props = [0.0 for _ in range(len(best_models))]
    recal_va_q_preds = [0.0 for _ in range(len(best_models))]
    recal_va_g_cali_scores = [0.0 for _ in range(len(best_models))]
    recal_va_scoring_rules = [0.0 for _ in range(len(best_models))]
    recal_te_cali_score = [0.0 for _ in range(len(best_models))]
    recal_te_sharp_score = [0.0 for _ in range(len(best_models))]
    recal_te_obs_props = [0.0 for _ in range(len(best_models))]
    recal_te_q_preds = [0.0 for _ in range(len(best_models))]
    recal_te_g_cali_scores = [0.0 for _ in range(len(best_models))]
    recal_te_scoring_rules = [0.0 for _ in range(len(best_models))]
    recal_model = [None for _ in range(len(best_models))]
    reduced_va_q_preds = [0.0 for _ in range(len(best_models))]
    for i, model in enumerate(best_models):
        if "calipso" in args.loss:
            #Calibrate remaining quantiles
            # tr_q_list = torch.linspace(0.01, 0.99, 99)
            tr_exp_props = torch.linspace(-0.01, 1.01, 103)
            tr_cali_score, tr_sharp_score, tr_obs_props, tr_q_preds, _, _ = test_uq(
                model,
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
        (
            va_cali_score[i],
            va_sharp_score[i],
            va_obs_props[i],
            va_q_preds[i],
            _,
            _
        ) = test_uq(
            model,
            x_va,
            y_va,
            va_exp_props,
            y_range,
            recal_model=recal_model_calipso,
            recal_type="sklearn",
        )
        reduced_va_q_preds[i] = va_q_preds[i][
            :, get_q_idx(va_exp_props, 0.01) : get_q_idx(va_exp_props, 0.99) + 1
        ]

        # Test UQ on test
        print("Testing UQ on test")
        te_exp_props = torch.linspace(0.01, 0.99, 99)
        (
            te_cali_score[i],
            te_sharp_score[i],
            te_obs_props[i],
            te_q_preds[i],
            te_g_cali_scores[i],
            te_scoring_rules[i],
        ) = test_uq(
            model,
            x_te,
            y_te,
            te_exp_props,
            y_range,
            recal_model=recal_model_calipso,
            recal_type="sklearn",
            test_group_cal=True,
        )

        # print('val', va_cali_score, va_sharp_score)
        print("\n")
        print("-" * 80)
        print(args.data)
        print("Draw frequency:", args.draw_group_every)
        print("Desired ECE:", desired_eces[i])
        print(
            "Val Cali: {:.3f}, Sharp: {:.3f}".format(va_cali_score[i], va_sharp_score[i])
        )
        print(
            "Test Cali: {:.3f}, Sharp: {:.3f}".format(te_cali_score[i], te_sharp_score[i])
        )
        print(te_g_cali_scores[i][:5])
        print(te_g_cali_scores[i][5:])
        print(te_scoring_rules[i])
        print("-" * 80)

        if args.recal:
            recal_model[i] = iso_recal(va_exp_props, va_obs_props[i])
            recal_exp_props = torch.linspace(0.01, 0.99, 99)

            (
                recal_va_cali_score[i],
                recal_va_sharp_score[i],
                recal_va_obs_props[i],
                recal_va_q_preds[i],
                recal_va_g_cali_scores[i],
                recal_va_scoring_rules[i]
            ) = test_uq(
                model,
                x_va,
                y_va,
                recal_exp_props,
                y_range,
                recal_model=recal_model[i],
                recal_type="sklearn",
                test_group_cal=True,
            )

            (
                recal_te_cali_score[i],
                recal_te_sharp_score[i],
                recal_te_obs_props[i],
                recal_te_q_preds[i],
                recal_te_g_cali_scores[i],
                recal_te_scoring_rules[i]
            ) = test_uq(
                model,
                x_te,
                y_te,
                recal_exp_props,
                y_range,
                recal_model=recal_model[i],
                recal_type="sklearn",
                test_group_cal=True,
            )
                
                # recal_model = iso_recal_ours(model_ens, x_va, y_va)
                # recal_va_cali_score2,recal_va_sharp_score2,_,_,_,_ = test_uq(
                #     model_ens,
                #     x_va,
                #     y_va,
                #     recal_exp_props,
                #     y_range,
                #     recal_model=recal_model[i],
                #     recal_type="sklearn",
                #     test_group_cal=True,
                # )
                # recal_te_cali_score2,recal_te_sharp_score2,_,_,_,_ = test_uq(
                #     model_ens,
                #     x_te,
                #     y_te,
                #     recal_exp_props,
                #     y_range,
                #     recal_model=recal_model[i],
                #     recal_type="sklearn",
                #     test_group_cal=True,
                # )
        print(
            "Recal Val Cali: {:.3f}, Sharp: {:.3f}".format(recal_va_cali_score[i], recal_va_sharp_score[i])
        )
        print(
            "Recal Test Cali: {:.3f}, Sharp: {:.3f}".format(recal_te_cali_score[i], recal_te_sharp_score[i])
        )
        # print(
        #     "Our Recal Val Cali: {:.3f}, Sharp: {:.3f}".format(recal_va_cali_score2, recal_va_sharp_score2)
        # )
        # print(
        #     "Our Recal Test Cali: {:.3f}, Sharp: {:.3f}".format(recal_te_cali_score2, recal_te_sharp_score2)
        # )

    save_dic = {
        "tr_loss_list": tr_loss_list,  # loss lists
        "va_loss_list": va_loss_list,
        "te_loss_list": te_loss_list,
        "va_cali_score": va_cali_score,  # test on va
        "va_sharp_score": va_sharp_score,
        "va_exp_props": va_exp_props,
        "va_obs_props": va_obs_props,
        "va_q_preds": va_q_preds,
        "te_cali_score": te_cali_score,  # test on te
        "te_sharp_score": te_sharp_score,
        "te_exp_props": te_exp_props,
        "te_obs_props": te_obs_props,
        "te_q_preds": te_q_preds,
        "te_g_cali_scores": te_g_cali_scores,
        "te_scoring_rules": te_scoring_rules,
        "recal_model": recal_model if args.recal else None,   # recalibration model
        "recal_exp_props": recal_exp_props if args.recal else None,
        "recal_va_cali_score": recal_va_cali_score if args.recal else None,
        "recal_va_sharp_score": recal_va_sharp_score if args.recal else None,
        "recal_va_obs_props": recal_va_obs_props if args.recal else None,
        "recal_va_q_preds": recal_va_q_preds if args.recal else None,
        "recal_va_g_cali_scores": recal_va_g_cali_scores if args.recal else None,
        "recal_va_scoring_rules": recal_va_scoring_rules if args.recal else None,
        "recal_te_cali_score": recal_te_cali_score if args.recal else None,
        "recal_te_sharp_score": recal_te_sharp_score if args.recal else None,
        "recal_te_obs_props": recal_te_obs_props if args.recal else None,
        "recal_te_q_preds": recal_te_q_preds if args.recal else None,
        "recal_te_g_cali_scores": recal_te_g_cali_scores if args.recal else None,
        "recal_te_scoring_rules": recal_te_scoring_rules if args.recal else None,
        "args": args,
        "model": model_ens,
        "best_models": best_models,
        "best_sharp_scores": best_sharp_scores,
        "desired_eces": desired_eces,
    }

    with open(save_file_name, "wb") as pf:
        pkl.dump(save_dic, pf)
