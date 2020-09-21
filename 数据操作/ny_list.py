import numpy as np
import pandas as pd
from tqdm import trange


def main():
    file_name = 'nyc_taxi_data.csv'
    data_pd = pd.read_csv(file_name)

    up_passengers = np.zeros((32, 32))
    off_passengers = np.zeros((32, 32))
    grid_index = np.zeros(32 * 32)

    for i in range(32 * 32):
        grid_index[i] = i + 1

    up_y = data_pd.up_y.values
    up_x = data_pd.up_x.values
    off_y = data_pd.off_y.values
    off_x = data_pd.off_x.values
    passengers = data_pd.passengers.values

    length = len(data_pd)
    for i in trange(length):
        up_passengers[int((1 - up_y[i]) * 32), int(up_x[i] * 32)] += passengers[i]
        off_passengers[int((1 - off_y[i]) * 32), int(off_x[i] * 32)] += passengers[i]

    up_passengers = up_passengers.reshape(-1)
    off_passengers = off_passengers.reshape(-1)
    data_all = np.zeros((1024, 3))

    for i in trange(1024):
        data_all[i, 0] = grid_index[i]
        data_all[i, 1] = up_passengers[i]
        data_all[i, 2] = off_passengers[i]

    data = pd.DataFrame(data_all, columns=["grid_index", "up_passengers", "off_passengers"])
    data.to_csv("nyc_taxi_grid_data.csv", index=False)
    
    return data_all


if __name__ == '__main__':
    data = main()
