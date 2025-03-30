import matplotlib.pyplot as plt
from matplotlib import rc
import torch

def plot_quantiles(X, Y, quantiles, printname = None):
    rc('font', size=26)
    # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    # rc('text', usetex=True)

    fig = plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')

    ax = fig.add_subplot()
    ax.set_axisbelow(True)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # SORT DATA
    _, sort = torch.sort(X, dim=0)
    sort = sort.reshape(X.shape[0], )
    plt.plot(X, Y, 'kx', markersize=14, label='Training Data', mew=2.0)
    plt.plot(X[sort], quantiles[sort].detach(), 'r')
    if not printname == None:
        plt.savefig('../figures/' + printname)

def plot_interval(X, Y, model, low_high=None, color ='C0', alpha=0.3, printname=None, ylim = None):

    if low_high is None:
        low_high = [0.025, 0.975]
    if ylim is None:
        deltaYmax = Y.max().item() - Y.min().item()
        ylim = [Y.min().item() - 0.1*deltaYmax, Y.max().item()+ 0.1*deltaYmax]

    rc('font', size=26)
    # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    # rc('text', usetex=True)

    fig = plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

    ax = fig.add_subplot()
    ax.set_axisbelow(True)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # SORT DATA
    _, sort = torch.sort(X, dim=0)
    sort = sort.reshape(X.shape[0], )
    n_x = X.shape[0]
    low_high = model.get_quantiles(X, low_high)
    plt.plot(X, Y, 'kx', markersize=14, label='Training Data', mew=2.0)
    plt.fill_between(X[sort].reshape(n_x,), low_high[sort].T[-1].detach().reshape(n_x,),
                     low_high[sort].T[0].detach().reshape(n_x,), color = color, alpha=alpha)
    plt.ylim(ylim)
    plt.xlabel(xlabel='x')
    plt.ylabel(ylabel='y')
    if not printname == None:
        plt.savefig('../figures/' + printname)


