import pandas as pd
import matplotlib.pyplot as plt

file_name = 'nyc_taxi_data.csv'
data_pd = pd.read_csv(file_name)

up_x = data_pd.up_x
up_y = data_pd.up_y
off_x = data_pd.off_x
off_y = data_pd.off_y
passengers = data_pd.passengers

print(data_pd.head())
print(data_pd.shape)
print(data_pd.isnull().sum())

fig = plt.figure(figsize=(20, 10))
alpha = 0.05

ax1 = fig.add_subplot(121)
plt.scatter(up_x, up_y, c='green', marker='.', alpha=alpha)
ax1.set_title('up location')

ax2 = fig.add_subplot(122)
plt.scatter(off_x, off_y, c='red', marker='.', alpha=alpha)
ax2.set_title('off location')

plt.show()
