from datasets.auto_mpg import *
from datasets.boston_house_prices import *
from datasets.yacht_hydrodynamics import *
from datasets.wine import *
from datasets.power import *
import pandas as pd
from scipy.io import arff

from sklearn.preprocessing import StandardScaler

def load_data(dataset_name, config_path = '/zfsauton2/home/acapone2/neurips2024/'):
    if dataset_name == 'mpg':
        return get_data_auto(config_path=config_path)
    elif dataset_name == 'boston':
        return get_boston_data(config_path=config_path)
    elif dataset_name == 'yacht':
        return get_yacht_data(config_path=config_path)
    elif dataset_name == 'wine':
        return get_wine_data(config_path=config_path)
    elif dataset_name == 'power':
        return get_power_data(config_path=config_path)
    else:
        raise NotImplementedError

    # # choose data set
    # print("Which experiment would you like to run? \n [T]oy problem, [B]oston house prices,"
    #        " [Y]acht, [W]ine, [FB2] Facebook Cooments 2, [M]PG, [K]in8nm or [C]oncrete")
    # while True:
    #     user_input = input()
    #     if user_input == "M":
    #         print("You have chosen to run the MPG experiment.")
    #         return get_data_auto(config_path=config_path)
    #     elif user_input == "B":
    #         print ("You have chosen to run the Boston house prices experiment.")
    #         return get_boston_data(config_path=config_path)
    #     # elif user_input == "Y":
    #     #     print ("You have chosen to run the yacht hydrodynamics experiment.")
    #     #     from datasets.yacht_hydrodynamics import *
    #     #     break
    #     # elif user_input == "W":
    #     #     print ("You have chosen to run the wine quality experiment.")
    #     #     from datasets.wine import *
    #     #     break
    #     # elif user_input == "FB2":
    #     #     print ("You have chosen to run the Facebook comments 2 experiment.")
    #     #     from datasets.facebook2_dataset import *
    #     #     break
    #     # elif user_input == "K":
    #     #     print("You have chosen to run the kin8nm experiment.")
    #     #     from datasets.kin8nm_dataset import *
    #     #     break
    #     # elif user_input == "C":
    #     #     print("You have chosen to run the concrete experiment.")
    #     #     from datasets.cement import *
    #     #     break
    #     else:
    #         print("Please enter T, B, M, Y, W, or S and press Enter.")
    #         continue

def load_data_bpl(dataset, seed, dataset_path='datasets/UCI_Datasets', extra_val = None):
    data = np.loadtxt("{}/{}.txt".format(dataset_path, dataset))
    x_al = data[:, :-1]
    y_al = data[:, -1].reshape(-1, 1)

    x_tr, x_te, y_tr, y_te = train_test_split(
        x_al, y_al, test_size=0.1, random_state=seed
    )
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tr, y_tr, test_size=0.2, random_state=seed
    )
    if extra_val is not None:
        x_tr, x_va2, y_tr, y_va2 = train_test_split(
            x_tr, y_tr, test_size=extra_val, random_state=seed
    )

    s_tr_x = StandardScaler().fit(x_tr)
    s_tr_y = StandardScaler().fit(y_tr)

    x_tr = torch.Tensor(s_tr_x.transform(x_tr))
    x_va = torch.Tensor(s_tr_x.transform(x_va))
    x_te = torch.Tensor(s_tr_x.transform(x_te))

    y_tr = torch.Tensor(s_tr_y.transform(y_tr))
    y_va = torch.Tensor(s_tr_y.transform(y_va))
    y_te = torch.Tensor(s_tr_y.transform(y_te))
    y_al = torch.Tensor(s_tr_y.transform(y_al))
    
    if extra_val is not None:
        x_va2 = torch.Tensor(s_tr_x.transform(x_va2))
        y_va2 = torch.Tensor(s_tr_y.transform(y_va2))
        return x_tr, x_va, x_va2, x_te, y_tr, y_va, y_va2, y_te, y_al

    return x_tr, x_va, x_te, y_tr, y_va, y_te, y_al


def load_data_mtr(dataset, seed, dataset_path='datasets/mtr_datasets', extra_val = None, start = 0, out_dim = 1):
    data = arff.loadarff(f'{dataset_path}/{dataset}.arff')
    df = pd.DataFrame(data[0])
    x_al = torch.tensor(df.iloc[:, start:-out_dim].to_numpy())
    y_al = torch.tensor(df.iloc[:, -out_dim:].to_numpy())

    x_tr, x_te, y_tr, y_te = train_test_split(
        x_al, y_al, test_size=0.15, random_state=seed
    )
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tr, y_tr, test_size=0.2/(1-0.15), random_state=seed
    )
    if extra_val is not None:
        x_tr, x_va2, y_tr, y_va2 = train_test_split(
            x_tr, y_tr, test_size=extra_val, random_state=seed
    )

    s_tr_x = StandardScaler().fit(x_tr)
    s_tr_y = StandardScaler().fit(y_tr)

    x_tr = torch.Tensor(s_tr_x.transform(x_tr))
    x_va = torch.Tensor(s_tr_x.transform(x_va))
    x_te = torch.Tensor(s_tr_x.transform(x_te))

    y_tr = torch.Tensor(s_tr_y.transform(y_tr))
    y_va = torch.Tensor(s_tr_y.transform(y_va))
    y_te = torch.Tensor(s_tr_y.transform(y_te))
    y_al = torch.Tensor(s_tr_y.transform(y_al))
    
    if extra_val is not None:
        x_va2 = torch.Tensor(s_tr_x.transform(x_va2))
        y_va2 = torch.Tensor(s_tr_y.transform(y_va2))
        return x_tr, x_va, x_va2, x_te, y_tr, y_va, y_va2, y_te, y_al

    return x_tr, x_va, x_te, y_tr, y_va, y_te, y_al

    
def load_data_csv(dataset, seed, dataset_path='/zfsauton2/home/acapone2/sharp_calibrated_models/datasets'):
    data = pd.read_csv("{}/{}.csv".format(dataset_path, dataset), index_col = 0).values
    # data = np.loadtxt("{}/{}.txt".format(dataset_path, dataset))
    x_al = data[:, :-1]
    y_al = data[:, -1].reshape(-1, 1)

    x_tr, x_te, y_tr, y_te = train_test_split(
        x_al, y_al, test_size=0.1, random_state=seed
    )
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tr, y_tr, test_size=0.2, random_state=seed
    )

    s_tr_x = StandardScaler().fit(x_tr)
    s_tr_y = StandardScaler().fit(y_tr)

    x_tr = torch.Tensor(s_tr_x.transform(x_tr))
    x_va = torch.Tensor(s_tr_x.transform(x_va))
    x_te = torch.Tensor(s_tr_x.transform(x_te))

    y_tr = torch.Tensor(s_tr_y.transform(y_tr))
    y_va = torch.Tensor(s_tr_y.transform(y_va))
    y_te = torch.Tensor(s_tr_y.transform(y_te))
    y_al = torch.Tensor(s_tr_y.transform(y_al))

    return x_tr, x_va, x_te, y_tr, y_va, y_te, y_al