import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DistributedSampler
from torchvision import transforms

def wind_data(path_name: str, random_state: int = 0, test_size:int=2*12*2):
    """
    Build the wind power data for the GEFcom IJF_paper case study.
    """

    df_wind = pd.read_csv(path_name, parse_dates=True, index_col=0)
    ZONES = ['ZONE_' + str(i) for i in range(1, 10 + 1)]

    # INPUTS DESCRIPTION
    # The predictors included wind forecasts at two heights, 10 and 100 m above ground level, obtained from the European Centre for Medium-range Weather Forecasts (ECMWF).
    # These forecasts were for the zonal and meridional wind components (denoted u and v), i.e., projections of the wind vector on the west-east and south-north axes, respectively.

    # U10 zonal wind component at 10 m
    # V10 meridional wind component at 10 m
    # U100 zonal wind component at 100 m
    # V100 meridional wind component at 100 m

    # ------------------------------------------------------------------------------------------------------------------
    # Build derived features
    # cf winner GEFcom2014 wind track “Probabilistic gradient boosting machines for GEFCom2014 wind forecasting”
    # ------------------------------------------------------------------------------------------------------------------

    # the wind speed (ws), wind energy (we), and wind direction (wd) were as follows,
    # where u and v are the wind components provided and d is the density, for which we used a constant 1.0
    # ws = sqrt[u**2  + v**2]
    # we = 0.5 × d × ws**3
    # wd = 180/π × arctan(u, v)

    df_wind['ws10'] = np.sqrt(df_wind['U10'].values ** 2 + df_wind['V10'].values ** 2)
    df_wind['ws100'] = np.sqrt(df_wind['U100'].values ** 2 + df_wind['V100'].values ** 2)
    df_wind['we10'] = 0.5 * 1 * df_wind['ws10'].values ** 3
    df_wind['we100'] = 0.5 * 1 * df_wind['ws100'].values ** 3
    df_wind['wd10'] = np.arctan2(df_wind['U10'].values, df_wind['V10'].values) * 180 / np.pi
    df_wind['wd100'] = np.arctan2(df_wind['U100'].values, df_wind['V100'].values) * 180 / np.pi

    features = ['U10', 'V10', 'U100', 'V100', 'ws10', 'ws100', 'we10', 'we100', 'wd10', 'wd100']

    data_zone = []
    for zone in ZONES:
        df_var = df_wind[df_wind[zone] == 1].copy()
        nb_days = int(len(df_var) / 24)
        zones = np.array([df_var[zone].values.reshape(nb_days, 24) for zone in ZONES])
        x = np.array([df_var[col].values.reshape(nb_days, 24) for col in features]) # (10, 731, 24)
        x = np.concatenate([x, zones], axis=0)  # (20, 731, 24)
        x = x.transpose(1, 2, 0).reshape(731, -1)
        # zones = [df_var[zone].values.reshape(nb_days, 24)[:, 0].reshape(nb_days, 1) for zone in ZONES]  # (10, 731, 1)
        # print(np.array(zones).shape, np.array(x).shape)
        # x = np.concatenate([df_var[col].values.reshape(nb_days, 24) for col in features] + zones, axis=1)
        # print(x[0], x[0].shape)

        y = df_var['TARGETVAR'].values.reshape(nb_days, 24)
        df_y = pd.DataFrame(data=y, index=df_var['TARGETVAR'].asfreq('D').index)
        df_x = pd.DataFrame(data=x, index=df_var['TARGETVAR'].asfreq('D').index)

        # Decomposition between LS, VS & TEST sets (TRAIN = LS + VS)
        df_x_train, df_x_TEST, df_y_train, df_y_TEST = train_test_split(df_x, df_y, test_size=test_size,random_state=random_state, shuffle=True)
        df_x_LS, df_x_VS, df_y_LS, df_y_VS = train_test_split(df_x_train, df_y_train, test_size=test_size,random_state=random_state, shuffle=True)

        data_zone.append([df_x_LS, df_y_LS, df_x_VS, df_y_VS, df_x_TEST, df_y_TEST])

        nb_days_LS = len(df_y_LS)
        nb_days_VS = len(df_y_VS)
        nb_days_TEST = len(df_y_TEST)
        print('#LS %s days #VS %s days # TEST %s days' % (nb_days_LS, nb_days_VS, nb_days_TEST))

    return [pd.concat([data_zone[i][j] for i in range(0, 9 + 1)], axis=0, join='inner') for j in range(0, 5 + 1)]

class WindDataDataset(Dataset):
    def __init__(self, df_x, df_y, scaler=None, x_scaler=None):
        super().__init__()
        """
        初始化数据集
        :param df_x: 标签数据
        :param df_y: 特征(条件)数据
        """
        self.df_x = df_x.values
        self.df_y = df_y
        print(self.df_x.shape, self.df_y.shape)
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.df_y)
        else:
            self.scaler = scaler
        self.df_y = self.scaler.transform(self.df_y)
        self.df_x = self.df_x.reshape(-1, 1, 24)
        self.df_y = self.df_y.reshape(-1, 20, 24)
        print(self.df_x.shape, self.df_y.shape)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本
        :param idx: 样本索引
        :return: 特征和标签
        """


        data = torch.tensor(self.df_x[idx], dtype=torch.float32)  # 将特征转为 Tensor conditions
        y = torch.tensor(self.df_y[idx], dtype=torch.float32)  # 将标签转为 Tensor  target
        # print(data.shape, y.shape)
        return data, y

    def __len__(self):
        """
        返回数据集的样本数量
        """
        return len(self.df_x)


def wind_data_zone2(path_name: str, random_state: int = 0, test_size:int=2*12*2):
    """
    Build the wind power data for the GEFcom IJF_paper case study.
    """

    df_wind = pd.read_csv(path_name, parse_dates=True, index_col=0)
    ZONES = ['ZONE_' + str(i) for i in range(1, 1 + 1)]

    # INPUTS DESCRIPTION
    # The predictors included wind forecasts at two heights, 10 and 100 m above ground level, obtained from the European Centre for Medium-range Weather Forecasts (ECMWF).
    # These forecasts were for the zonal and meridional wind components (denoted u and v), i.e., projections of the wind vector on the west-east and south-north axes, respectively.

    # U10 zonal wind component at 10 m
    # V10 meridional wind component at 10 m
    # U100 zonal wind component at 100 m
    # V100 meridional wind component at 100 m

    # ------------------------------------------------------------------------------------------------------------------
    # Build derived features
    # cf winner GEFcom2014 wind track “Probabilistic gradient boosting machines for GEFCom2014 wind forecasting”
    # ------------------------------------------------------------------------------------------------------------------

    # the wind speed (ws), wind energy (we), and wind direction (wd) were as follows,
    # where u and v are the wind components provided and d is the density, for which we used a constant 1.0
    # ws = sqrt[u**2  + v**2]
    # we = 0.5 × d × ws**3
    # wd = 180/π × arctan(u, v)

    df_wind['ws10'] = np.sqrt(df_wind['U10'].values ** 2 + df_wind['V10'].values ** 2)
    df_wind['ws100'] = np.sqrt(df_wind['U100'].values ** 2 + df_wind['V100'].values ** 2)
    df_wind['we10'] = 0.5 * 1 * df_wind['ws10'].values ** 3
    df_wind['we100'] = 0.5 * 1 * df_wind['ws100'].values ** 3
    df_wind['wd10'] = np.arctan2(df_wind['U10'].values, df_wind['V10'].values) * 180 / np.pi
    df_wind['wd100'] = np.arctan2(df_wind['U100'].values, df_wind['V100'].values) * 180 / np.pi

    features = ['U10', 'V10', 'U100', 'V100', 'ws10', 'ws100', 'we10', 'we100', 'wd10', 'wd100']

    data_zone = []
    for zone in ZONES:
        df_var = df_wind.copy()
        nb_days = int(len(df_var) / 24)
        # zones = np.array([df_var[zone].values.reshape(nb_days, 24) for zone in ZONES])
        x = np.array([df_var[col].values.reshape(nb_days, 24) for col in features]) # (10, 731, 24)
        # x = np.concatenate([x, zones], axis=0)  # (20, 731, 24)
        x = x.transpose(1, 2, 0).reshape(731, -1)
        # zones = [df_var[zone].values.reshape(nb_days, 24)[:, 0].reshape(nb_days, 1) for zone in ZONES]  # (10, 731, 1)
        # print(np.array(zones).shape, np.array(x).shape)
        # x = np.concatenate([df_var[col].values.reshape(nb_days, 24) for col in features] + zones, axis=1)
        # print(x[0], x[0].shape)

        y = df_var['TARGETVAR'].values.reshape(nb_days, 24)
        df_y = pd.DataFrame(data=y, index=df_var['TARGETVAR'].asfreq('D').index)
        df_x = pd.DataFrame(data=x, index=df_var['TARGETVAR'].asfreq('D').index)

        # Decomposition between LS, VS & TEST sets (TRAIN = LS + VS)
        df_x_train, df_x_TEST, df_y_train, df_y_TEST = train_test_split(df_x, df_y, test_size=test_size,random_state=random_state, shuffle=True)
        df_x_LS, df_x_VS, df_y_LS, df_y_VS = train_test_split(df_x_train, df_y_train, test_size=test_size,random_state=random_state, shuffle=True)

        data_zone.append([df_x_LS, df_y_LS, df_x_VS, df_y_VS, df_x_TEST, df_y_TEST])

        nb_days_LS = len(df_y_LS)
        nb_days_VS = len(df_y_VS)
        nb_days_TEST = len(df_y_TEST)
        print('#LS %s days #VS %s days # TEST %s days' % (nb_days_LS, nb_days_VS, nb_days_TEST))

    return [pd.concat([data_zone[i][j] for i in range(0, 1)], axis=0, join='inner') for j in range(0, 5 + 1)]

class WindDataDataset_zone2(Dataset):
    def __init__(self, df_x, df_y, scaler=None, x_scaler=None):
        super().__init__()
        """
        初始化数据集
        :param df_x: 标签数据
        :param df_y: 特征(条件)数据
        """
        self.df_x = df_x.values
        self.df_y = df_y
        print(self.df_x.shape, self.df_y.shape)
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.df_y)
        else:
            self.scaler = scaler
        self.df_y = self.scaler.transform(self.df_y)
        self.df_x = self.df_x.reshape(-1, 1, 24)
        self.df_y = self.df_y.reshape(-1, 10, 24)
        print(self.df_x.shape, self.df_y.shape)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本
        :param idx: 样本索引
        :return: 特征和标签
        """

        data = torch.tensor(self.df_x[idx], dtype=torch.float32)  # 将特征转为 Tensor conditions
        y = torch.tensor(self.df_y[idx], dtype=torch.float32)  # 将标签转为 Tensor  target
        # print(data.shape, y.shape)
        return data, y

    def __len__(self):
        """
        返回数据集的样本数量
        """
        return len(self.df_x)

if __name__ == '__main__':

    # prepare data
    data = wind_data(path_name="../scenarios/wind_data_all_zone.csv", test_size=50, random_state=0)

    df_x_LS = data[0].copy()    # x 是标签
    df_y_LS = data[1].copy()    # y 是特征

    nb_days_LS = len(df_y_LS)
    print(f"{df_x_LS.shape}, {df_y_LS.shape}")
    print('all zones #LS %s days' % (nb_days_LS))

    dataset = WindDataDataset(df_y_LS, df_x_LS)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    count = 0
    for data, y in dataloader:
        print(data.shape)
        # count += 1
        # print(count)
        print(y.shape)
        break


    # prepare data
    data = wind_data_zone2(path_name="../scenarios/wind_zone2.csv", test_size=50, random_state=0)

    df_x_LS = data[0].copy()    # x 是标签
    df_y_LS = data[1].copy()    # y 是特征

    nb_days_LS = len(df_y_LS)
    print(f"{df_x_LS.shape}, {df_y_LS.shape}")
    print('all zones #LS %s days' % (nb_days_LS))

    dataset = WindDataDataset_zone2(df_y_LS, df_x_LS)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    count = 0
    for data, y in dataloader:
        print(data.shape)
        # count += 1
        # print(count)
        print(y.shape)
        break