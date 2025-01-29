import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt
import sys
import tqdm
sys.path.append('/home/kzaidi/sharp_calibrated_models/')
from models.quantile_ensemble import *
from auxiliary.plot_quantiles import *
from auxiliary.metrics import *
from models.bayesian_nn import *
from models.deshpande_kuleshov_cal import *
from models.kuleshov_2018_cal import *
from auxiliary.load_data import load_data, load_data_bpl, load_data_csv
import copy

def gen_model(dataset, seed, path, output_device, extra_val=None):
    #seed = 0
    # if torch.cuda.is_available():
    #     output_device = 'cuda'
    # else:
    #     output_device = 'cpu'
    #dataset = 'boston'
    #dataset = 'Boston'
    val = True
    nepochs = 1000
    display = True

    np.random.seed(seed)
    torch.manual_seed(seed)

    if extra_val is None:
        X, X_val, X_test, Y, Y_val, Y_test, Y_al = load_data_bpl(dataset, seed, dataset_path='/home/kzaidi/calibrated-quantile-uq/data/UCI_Datasets')
    else:
        X, X_val, X_true_val, X_test, Y, Y_val, Y_true_val, Y_test, Y_al = load_data_bpl(dataset, seed, extra_val=extra_val)
    #X, X_val, X_test, Y, Y_val, Y_test, Y_al = load_data_csv(dataset, seed)

    X = X.to(output_device)
    Y = Y.to(output_device)
    X_val = X_val.to(output_device)
    Y_val = Y_val.to(output_device)
    X_test = X_test.to(output_device)
    Y_test = Y_test.to(output_device)
    if not val:
        X=torch.cat((X, X_val), dim=0)
        Y=torch.cat((Y, Y_val), dim=0)

    class data_set(Dataset):
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, index):
            return self.X[index], self.Y[index]

    data = data_set(X=X, Y=Y)
    dataloader = DataLoader(data, batch_size=64, shuffle=True)#25

    our_model = bpl_nn(X.shape[1]).to(output_device)
    optimizer = optim.Adam(our_model.parameters(), lr=1e-3)
    loss_fun = torch.nn.MSELoss()

    best_loss = torch.inf
    best_weights = None
    early_stop_count = 0

    training_metrics = {'tr': [], 'va': []}

    for epoch in (tqdm(range(nepochs)) if display else range(nepochs)):
        our_model.train()
        batch_loss = []
        for Xbatch, Ybatch in dataloader:
            Xbatch, Ybatch = Xbatch.to(output_device), Ybatch.to(output_device)
            pred = our_model(Xbatch)
            loss = loss_fun(pred,Ybatch)
            batch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        training_metrics['tr'].append(np.mean(batch_loss))
        
        if val:
            our_model.eval()
            with torch.no_grad():
                Xbatch, Ybatch = X_val.to(output_device), Y_val.to(output_device)
                pred = our_model(Xbatch)
                loss = loss_fun(pred,Ybatch)
                if loss < best_loss:
                    early_stop_count = 0
                    best_loss = loss
                    best_weights = copy.deepcopy(our_model.state_dict())
                else:
                    early_stop_count += 1
            if early_stop_count > 200:
                break
            training_metrics['va'].append(loss.item())
            
    if val:
        our_model.load_state_dict(best_weights)

    #torch.save(our_model.state_dict(), path)
    with open(path, 'wb') as pf:
        import pickle as pkl
        pkl.dump(our_model, pf)

    # plt.plot(training_metrics['tr'])
    # plt.plot(training_metrics['va'])
    # plt.savefig('bla')
    # plt.show()
    return copy.deepcopy(our_model.state_dict())

if __name__ == "__main__":
    gen_model('boston',0,'base_reg_vanilla_model.pt','cuda')