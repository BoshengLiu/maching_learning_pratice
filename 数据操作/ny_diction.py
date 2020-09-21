import numpy as np
import pandas as pd
from tqdm import trange


def main():
    file_name = 'nyc_taxi_data.csv'
    data_pd = pd.read_csv(file_name)

    diction = {}
    up_y = data_pd.up_y.values
    up_x = data_pd.up_x.values
    off_y = data_pd.off_y.values
    off_x = data_pd.off_x.values
    passengers = data_pd.passengers.values

    length = len(data_pd)
    for i in trange(length):
        x1, y1 = (int((1 - up_y[i]) * 32), int(up_x[i] * 32))
        x2, y2 = (int((1 - off_y[i]) * 32), int(off_x[i] * 32))
        diction.setdefault((x1, y1), [0, 0])
        diction.setdefault((x2, y2), [0, 0])
        diction[(x1, y1)][0] += passengers[i]
        diction[(x2, y2)][1] += passengers[i]

    data_all = np.zeros((1024, 3))
    for k, v in diction.items():
        position = k[0] * 32 + k[1]
        data_all[position, 0] = position + 1
        data_all[position, 1:] = v

    data = pd.DataFrame(data_all, columns=["grid_index", "up_passengers",
                                           "off_passengers"])
    data.to_csv("nyc_taxi_grid_data.csv", index=False)
    return data_all


if __name__ == '__main__':
    data = main()
    file_name = 'nyc_taxi_grid_data.csv'
    data_pd = pd.read_csv(file_name)
    index = data_pd['grid_index']
    for i in range(len(data_pd)):
        index[i] = i + 1
    data_pd.to_csv("nyc_taxi_grid_data_dict.csv", index=False)
