import os, sys
from copy import deepcopy
import tqdm
import numpy as np
import torch
from typing import Optional
from utils.q_model_ens import uq_model, gather_loss_per_q, get_ens_pred_interp, get_ens_pred_conf_bound

# sys.path.append('../utils/NNKit')
# sys.path.append('utils')
from scipy.stats import norm as norm_distr
from scipy.stats import t as t_distr
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from NNKit.models.model import vanilla_nn, bpl_nn, standard_nn_model

""" QModelEns Utils """


def gather_loss_per_q(loss_fn, model, y, x, q_list, device, args):
    loss_list = []
    for q in q_list:
        q_loss = loss_fn(model, y, x, q, device, args)
        loss_list.append(q_loss)
    loss = torch.mean(torch.stack(loss_list))

    return loss


def get_ens_pred_interp(unc_preds, taus, fidelity=10000):
    """
    unc_preds 3D ndarray (ens_size, 99, num_x)
    where for each ens_member, each row corresonds to tau 0.01, 0.02...
    and the columns are for the set of x being predicted over.
    """
    # taus = np.arange(0.01, 1, 0.01)
    y_min, y_max = np.min(unc_preds), np.max(unc_preds)
    y_grid = np.linspace(y_min, y_max, fidelity)
    new_quants = []
    avg_cdfs = []
    for x_idx in tqdm.tqdm(range(unc_preds.shape[-1])):
        x_cdf = []
        for ens_idx in range(unc_preds.shape[0]):
            xs, ys = [], []
            targets = unc_preds[ens_idx, :, x_idx]
            for idx in np.argsort(targets):
                if len(xs) != 0 and targets[idx] <= xs[-1]:
                    continue
                xs.append(targets[idx])
                ys.append(taus[idx])
            intr = interp1d(
                xs, ys, kind="linear", fill_value=([0], [1]), bounds_error=False
            )
            x_cdf.append(intr(y_grid))
        x_cdf = np.asarray(x_cdf)
        avg_cdf = np.mean(x_cdf, axis=0)
        avg_cdfs.append(avg_cdf)
        t_idx = 0
        x_quants = []
        for idx in range(len(avg_cdf)):
            if t_idx >= len(taus):
                break
            if taus[t_idx] <= avg_cdf[idx]:
                x_quants.append(y_grid[idx])
                t_idx += 1
        while t_idx < len(taus):
            x_quants.append(y_grid[-1])
            t_idx += 1
        new_quants.append(x_quants)
    return np.asarray(new_quants).T


def get_ens_pred_conf_bound(unc_preds, taus, conf_level=0.95, score_distr="z"):
    """
    unc_preds 3D ndarray (ens_size, num_tau, num_x)
    where for each ens_member, each row corresonds to tau 0.01, 0.02...
    and the columns are for the set of x being predicted over.
    """
    num_ens, num_tau, num_x = unc_preds.shape
    len_tau = taus.size

    mean_pred = np.mean(unc_preds, axis=0)
    std_pred = np.std(unc_preds, axis=0, ddof=1)
    stderr_pred = std_pred / np.sqrt(num_ens)
    alpha = 1 - conf_level  # is (1-C)

    # determine coefficient
    if score_distr == "z":
        crit_value = norm_distr.ppf(1 - (0.5 * alpha))
    elif score_distr == "t":
        crit_value = t_distr.ppf(q=1 - (0.5 * alpha), df=(num_ens - 1))
    else:
        raise ValueError("score_distr must be one of z or t")

    gt_med = (taus > 0.5).reshape(-1, num_x)
    lt_med = ~gt_med
    assert gt_med.shape == mean_pred.shape == stderr_pred.shape
    out = (
        lt_med * (mean_pred - (float(crit_value) * stderr_pred))
        + gt_med * (mean_pred + (float(crit_value) * stderr_pred))
    ).T
    out = torch.from_numpy(out)
    return out


class CalipsoModel(uq_model):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers,
        lr,
        wd,
        num_ens,
        device,
        big_model = False,
        num_quantiles = 6,
        cali_X = None,
        cali_y = None,
        pretrain = False,
    ):
        assert num_quantiles % 2 == 0, "num_quantiles must be even"
        self.num_ens = num_ens
        self.num_quantiles = num_quantiles
        self.device = device
        self.quantile_models = []
        for _ in range(num_quantiles):
            #Changed for ICML Rebuttal!
            if big_model:
                quantile_model = standard_nn_model(nfeatures=input_size).to(device)
            else:            
                quantile_model = vanilla_nn( #Changed for ICML Rebuttal!
                    input_size=input_size,
                    output_size=output_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                ).to(device)
            if pretrain:
                train_X = cali_X.to(device)
                train_y = cali_y.to(device)
                optimizer = torch.optim.Adam(quantile_model.parameters(), lr=0.01)
                loss_fn = torch.nn.MSELoss().to(device)
                for _ in range(100):
                    optimizer.zero_grad()
                    output = quantile_model(train_X)
                    loss = loss_fn(output, train_y)
                    loss.backward()
                    optimizer.step()
            self.quantile_models.append(quantile_model)
                
            
        self.optimizers = [
            torch.optim.Adam(x.parameters(), lr=lr, weight_decay=wd)
            for x in self.quantile_models
        ]
        self.keep_training = [True for _ in range(num_quantiles)]
        self.best_va_loss = [np.inf for _ in range(num_quantiles)]
        self.best_va_model = [None for _ in range(num_quantiles)]
        self.best_va_ep = [0 for _ in range(num_quantiles)]
        self.done_training = False
        self.cali_datasets = [(cali_X.to(device), cali_y.to(device)) for _ in range(num_quantiles//2)]
        self.alphas = [None for _ in range(num_quantiles-2)]


    def use_device(self, device):
        self.device = device
        for idx in range(len(self.quantile_models)//2):
            self.quantile_models[idx] = self.quantile_models[idx].to(device)
            self.quantile_models[-1-idx] = self.quantile_models[-1-idx].to(device)
            self.cali_datasets[idx] = (self.cali_datasets[idx][0].to(device), self.cali_datasets[idx][1].to(device))

        if device.type == "cuda":
            for idx in range(len(self.quantile_models)):
                assert next(self.quantile_models[idx].parameters()).is_cuda

    def print_device(self):
        device_list = []
        for idx in range(len(self.quantile_models)):
            if next(self.quantile_models[idx].parameters()).is_cuda:
                device_list.append("cuda")
            else:
                device_list.append("cpu")
        print(device_list)

    def loss(self, dummy_loss_fn, x, y, q_list, batch_q, take_step, args):
        self.update_datasets()
        quantile_loss = []
        num_quantile_pairs = self.num_quantiles // 2
        for idx in range(num_quantile_pairs):
            self.optimizers[idx].zero_grad()
            self.optimizers[-1-idx].zero_grad()
            if self.keep_training[idx]:
                c_lower, c_upper = self.get_clower_cupper(idx)
                loss_upper = torch.mean(torch.abs(self.quantile_models[-1-idx](x)+c_upper - y))
                loss_lower = torch.mean(torch.abs(self.quantile_models[idx](x)+c_lower - y))
                loss = loss_upper + loss_lower

                # if batch_q:
                #     loss = calipso_loss(
                #         self.quantile_models[idx], Y, X, q_list, self.device, args
                #     )
                # else:
                #     loss = gather_loss_per_q(
                #         calipso_loss,
                #         self.quantile_models[idx],
                #         y,
                #         x,
                #         q_list,
                #         self.device,
                #         args,
                #     )
                quantile_loss.append(loss.item())

                if take_step:
                    loss.backward()
                    self.optimizers[idx].step()
                    self.optimizers[-1-idx].step()

            else:
                quantile_loss.append(np.nan)

        return np.asarray(quantile_loss)
    
    def get_clower_cupper(self, idx):
        X_cali, Y_cali = self.cali_datasets[idx]
        q_upper = self.quantile_models[-1-idx](X_cali)
        q_lower = self.quantile_models[idx](X_cali)
        c_upper = (Y_cali-q_upper).max()
        c_lower = (Y_cali-q_lower).min()
        return c_lower-1e-3, c_upper+1e-3
    
    def update_datasets(self, idx = Optional[None]):
        quantiles = torch.linspace(0, 1, self.num_quantiles)
        X_cali_total, Y_cali_total = self.cali_datasets[0]
        q_lower_prev, q_upper_prev = None, None

        for i in range(self.num_quantiles//2-1):
            c_lower, c_upper = self.get_clower_cupper(i)
            q_lower = self.quantile_models[i](X_cali_total) + c_lower
            q_upper = self.quantile_models[-1-i](X_cali_total) + c_upper
            if q_lower_prev is not None and q_upper_prev is not None:
                q_upper = torch.min(q_upper, q_upper_prev)
                q_lower = torch.max(q_lower, q_lower_prev)
            dq = q_upper - q_lower
            delta = quantiles[i+1]
            alpha_lower = torch.quantile((Y_cali_total - q_lower)/dq, delta.item())
            alpha_upper = torch.quantile((q_upper - Y_cali_total)/dq, delta.item())
            q_lower_prev = q_lower + alpha_lower*dq
            q_upper_prev = q_upper - alpha_upper*dq
            self.alphas[i] = alpha_lower
            self.alphas[-1-i] = alpha_upper
            ind_keep = ((q_lower_prev <= Y_cali_total) & (Y_cali_total <= q_upper_prev)).squeeze()
            self.cali_datasets[i+1] = (X_cali_total[ind_keep], Y_cali_total[ind_keep])
        
        

    def update_va_loss(
        self, loss_fn, x, y, q_list, batch_q, curr_ep, num_wait, args
    ):
        with torch.no_grad():
            va_loss = self.loss(
                loss_fn, x, y, q_list, batch_q, take_step=False, args=args
            )

        for idx in range(self.num_ens):
            if self.keep_training[idx]:
                if va_loss[idx] < self.best_va_loss[idx]:
                    self.best_va_loss[idx] = va_loss[idx]
                    self.best_va_ep[idx] = curr_ep
                    self.best_va_model[idx] = deepcopy(self.quantile_models[idx])
                else:
                    if curr_ep - self.best_va_ep[idx] > num_wait:
                        print(
                            "Val loss stagnate for {}, model {}".format(
                                num_wait, idx
                            )
                        )
                        print("EP {}".format(curr_ep))
                        self.keep_training[idx] = False

        if not any(self.keep_training):
            self.done_training = True

        return va_loss

    #####
    def predict(
        self,
        cdf_in,
        conf_level=0.95,
        score_distr="z",
        recal_model=None,
        recal_type=None,
    ):
        """
        Only pass in cdf_in into model and return output
        If self is an ensemble, return a conservative output based on conf_bound
        specified by conf_level

        :param cdf_in: tensor [x, p], of size (num_x, dim_x + 1)
        :param conf_level: confidence level for ensemble prediction
        :param score_distr: 'z' or 't' for confidence bound coefficient
        :param recal_model:
        :param recal_type:
        :return:
        """

        if self.num_ens == 1:
            with torch.no_grad():
                pred = self.best_va_model[0](cdf_in)
        if self.num_ens > 1:
            pred_list = []
            for m in self.best_va_model:
                with torch.no_grad():
                    pred_list.append(m(cdf_in).T.unsqueeze(0))

            unc_preds = (
                torch.cat(pred_list, dim=0).detach().cpu().numpy()
            )  # shape (num_ens, num_x, 1)
            taus = cdf_in[:, -1].flatten().cpu().numpy()
            pred = get_ens_pred_conf_bound(
                unc_preds, taus, conf_level=0.95, score_distr="z"
            )
            pred = pred.to(cdf_in.device)

        return pred

    #####

    def predict_q(
        self,
        x,
        q_list=None,
        ens_pred_type="conf",
        recal_model=None,
        recal_type=None,
    ):
        """
        Get output for given list of quantiles

        :param x: tensor, of size (num_x, dim_x)
        :param q_list: flat tensor of quantiles, if None, is set to [0.01, ..., 0.99]
        :param ens_pred_type:
        :param recal_model:
        :param recal_type:
        :return:
        """
        self.update_datasets()
        if q_list is None:
            q_list = torch.arange(0.01, 0.99, 0.01)
        else:
            q_list = q_list.flatten()
        # Rearrange q_list to be between 0 and 1
        q_list = q_list.clip(0, 1)

        num_x = x.shape[0]
        num_q = q_list.shape[0]

        cdf_preds = []

        quantiles = [[] for _ in range(self.num_quantiles)]
        q_lower_prev, q_upper_prev = None, None
        for i in range(self.num_quantiles//2):
            c_lower, c_upper = self.get_clower_cupper(i)
            q_lower = self.quantile_models[i](x) + c_lower
            q_upper = self.quantile_models[-1-i](x) + c_upper
            if q_lower_prev is not None and q_upper_prev is not None:
                q_upper = torch.min(q_upper, q_upper_prev)
                q_lower = torch.max(q_lower, q_lower_prev)
            dq = q_upper - q_lower
            quantiles[i] = q_lower
            quantiles[-1-i] = q_upper
            if i < self.num_quantiles//2-1:
                q_lower_prev = q_lower + self.alphas[i]*dq
                q_upper_prev = q_upper - self.alphas[-1-i]*dq
        quantiles = torch.cat(quantiles, dim=1)

        for p in q_list:
            if recal_model is not None:
                if recal_type == "torch":
                    recal_model.cpu()  # keep recal model on cpu
                    with torch.no_grad():
                        in_p = recal_model(p.reshape(1, -1)).item()
                elif recal_type == "sklearn":
                    in_p = float(recal_model.predict(p.flatten()))
                else:
                    raise ValueError("recal_type incorrect")
            else:
                in_p = float(p)
            cdf_pred = torch.quantile(quantiles, p, dim=1).reshape(num_x, 1)
            # p_tensor = (in_p * torch.ones(num_x)).reshape(-1, 1)
            # cdf_in = torch.cat([x, p_tensor], dim=1).to(self.device)
            # cdf_pred = self.predict(cdf_in)  # shape (num_x, 1)
            cdf_preds.append(cdf_pred)

        pred_mat = torch.cat(cdf_preds, dim=1)  # shape (num_x, num_q)
        assert pred_mat.shape == (num_x, num_q)
        return pred_mat



        # ###
        # cdf_preds = []
        # for p in q_list:
        #     if recal_model is not None:
        #         if recal_type == 'torch':
        #             recal_model.cpu() # keep recal model on cpu
        #             with torch.no_grad():
        #                 in_p = recal_model(p.reshape(1, -1)).item()
        #         elif recal_type == 'sklearn':
        #             in_p = float(recal_model.predict(p.flatten()))
        #         else:
        #             raise ValueError('recal_type incorrect')
        #     else:
        #         in_p = float(p)
        #     p_tensor = (in_p * torch.ones(num_pts)).reshape(-1, 1)
        #     cdf_in = torch.cat([x, p_tensor], dim=1).to(self.device)
        #
        #     ens_preds_p = []
        #     with torch.no_grad():
        #         for m in self.best_va_model:
        #             cdf_pred = m(cdf_in).reshape(num_pts, -1)
        #             ens_preds_p.append(cdf_pred.flatten())
        #
        #     cdf_preds.append(torch.stack(ens_preds_p, dim=0).unsqueeze(1))
        # ens_pred_mat = torch.cat(cdf_preds, dim=1).numpy()
        #
        # if self.num_ens > 1:
        #     assert ens_pred_mat.shape == (self.num_ens, num_q, num_pts)
        #     ens_pred = ens_pred_fn(ens_pred_mat, taus=q_list)
        # else:
        #     ens_pred = ens_pred_mat.reshape(num_q, num_pts)
        # return ens_pred
        # ###


    
if __name__ == "__main__":
    temp_model = QModelEns(
        input_size=1,
        output_size=1,
        hidden_size=10,
        num_layers=2,
        lr=0.01,
        wd=0.0,
        num_ens=5,
        device=torch.device("cuda:0"),
    )
